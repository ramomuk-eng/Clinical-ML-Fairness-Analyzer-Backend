import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fairness_metrics import equalized_odds, classification_metrics

def run_baseline(X, y, race):
    X_train, X_test, y_train, y_test, race_train, race_test = \
        train_test_split(X, y, race,
                        test_size=0.2,
                        random_state=42,
                        stratify=y)

    model = GradientBoostingClassifier(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.1,
    random_state=42,
    subsample=0.5
)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Find threshold for ~0.9 sensitivity
    threshold = find_threshold(y_test, y_prob, target_sensitivity=0.75)
    y_pred = (y_prob >= threshold).astype(int)

    clf_metrics = classification_metrics(y_test, y_pred, y_prob)
    fairness = equalized_odds(y_test, y_pred, race_test)

    return clf_metrics, fairness, X_test, y_test, race_test, y_prob

def find_threshold(y_true, y_prob, target_sensitivity=0.75):
    best_threshold = 0.5
    best_diff = float('inf')

    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        diff = abs(sens - target_sensitivity)
        if diff < best_diff:
            best_diff = diff
            best_threshold = thresh

    return best_threshold


class Predictor(nn.Module):
    def __init__(self, input_dim):
        super(Predictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class Adversary(nn.Module):
    def __init__(self, num_groups):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, num_groups),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)


def run_adversarial(X, y, race, alpha=1.0, epochs=50):
    le = LabelEncoder()
    race_encoded = le.fit_transform(race)
    num_groups = len(le.classes_)

    X_train, X_test, y_train, y_test, race_train, race_test = \
        train_test_split(X, y, race_encoded,
                        test_size=0.2,
                        random_state=42,
                        stratify=y)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    race_train_t = torch.LongTensor(race_train)

    X_test_t = torch.FloatTensor(X_test)

    predictor = Predictor(X_train.shape[1])
    adversary = Adversary(num_groups)

    pred_optimizer = optim.Adam(predictor.parameters(), lr=0.001)
    adv_optimizer = optim.Adam(adversary.parameters(), lr=0.001)

    pred_criterion = nn.BCELoss()
    adv_criterion = nn.CrossEntropyLoss()

    batch_size = 256

    for epoch in range(epochs):
        predictor.train()
        adversary.train()

        perm = torch.randperm(X_train_t.size(0))

        for i in range(0, X_train_t.size(0), batch_size):
            idx = perm[i:i + batch_size]
            X_batch = X_train_t[idx]
            y_batch = y_train_t[idx]
            race_batch = race_train_t[idx]

            # Train adversary
            pred_out = predictor(X_batch).detach()
            adv_out = adversary(pred_out)
            adv_loss = adv_criterion(adv_out, race_batch)

            adv_optimizer.zero_grad()
            adv_loss.backward()
            adv_optimizer.step()

            # Train predictor
            pred_out = predictor(X_batch)
            adv_out = adversary(pred_out)

            pred_loss = pred_criterion(pred_out, y_batch)
            adv_loss_for_pred = adv_criterion(adv_out, race_batch)

            # Predictor wants to minimize pred_loss
            # and maximize adversary loss (fool the adversary)
            total_loss = pred_loss - alpha * adv_loss_for_pred

            pred_optimizer.zero_grad()
            total_loss.backward()
            pred_optimizer.step()

    # Evaluate
    predictor.eval()
    with torch.no_grad():
        y_prob = predictor(X_test_t).numpy().flatten()

    race_test_labels = le.inverse_transform(race_test)
    threshold = find_threshold(y_test, y_prob, target_sensitivity=0.75)
    y_pred = (y_prob >= threshold).astype(int)

    clf_metrics = classification_metrics(y_test, y_pred, y_prob)
    fairness = equalized_odds(y_test, y_pred, race_test_labels)

    return clf_metrics, fairness