"""
Minimal TransE-style knowledge-graph embedding model, trained directly
on labelled (true/false) triples with a logistic (BCE) loss instead of
the usual random-corruption margin loss -- since here we actually have
*real* negative examples (facts explicitly marked false), which is a
much stronger supervision signal than synthetic corruption.

score(s,p,o) = b_p - || e_s + r_p - e_o ||_2     (higher => more plausible)
p(true)      = sigmoid(score)

b_p is a learnable per-relation bias/margin. Without it, the score is
always <= 0 (a norm can't be negative), which caps sigmoid(score) at 0.5
and makes it impossible for the model to ever confidently predict "true".
The bias lets the model learn, per relation, how large a distance is
still "close enough" to count as plausible.
"""
import numpy as np

class TransEClassifier:
    def __init__(self, dim=32, lr=0.2, epochs=500, l2=1e-3, seed=0):
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed

    def fit(self, n_entities, n_relations, s_idx, p_idx, o_idx, y, verbose=False):
        rng = np.random.default_rng(self.seed)
        d = self.dim
        self.E = rng.normal(0, 0.5, size=(n_entities, d))
        self.R = rng.normal(0, 0.1, size=(n_relations, d))
        self.b = np.zeros(n_relations)
        s_idx = np.asarray(s_idx); p_idx = np.asarray(p_idx)
        o_idx = np.asarray(o_idx); y = np.asarray(y, dtype=float)
        n = len(y)

        for epoch in range(self.epochs):
            d_vec = self.E[s_idx] + self.R[p_idx] - self.E[o_idx]        # (n, dim)
            norm = np.linalg.norm(d_vec, axis=1) + 1e-9                  # (n,)
            score = self.b[p_idx] - norm
            p = 1 / (1 + np.exp(-score))

            g_score = p - y                       # dL/dscore  (BCE + sigmoid identity)
            g_norm = -g_score                      # dscore/dnorm = -1
            grad_d = (g_norm / norm)[:, None] * d_vec   # dL/dd

            gE = np.zeros_like(self.E); gR = np.zeros_like(self.R); gb = np.zeros_like(self.b)
            np.add.at(gE, s_idx, grad_d)
            np.add.at(gE, o_idx, -grad_d)
            np.add.at(gR, p_idx, grad_d)
            np.add.at(gb, p_idx, g_score)

            self.E -= self.lr * (gE / n + self.l2 * self.E)
            self.R -= self.lr * (gR / n + self.l2 * self.R)
            self.b -= self.lr * (gb / n)

            if verbose and epoch % max(1, self.epochs // 10) == 0:
                loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
                acc = ((p > 0.5).astype(float) == y).mean()
                print(f"epoch {epoch:4d}  loss {loss:.4f}  acc {acc:.4f}")
        return self

    def score(self, s_idx, p_idx, o_idx):
        s_idx = np.asarray(s_idx); p_idx = np.asarray(p_idx); o_idx = np.asarray(o_idx)
        d_vec = self.E[s_idx] + self.R[p_idx] - self.E[o_idx]
        norm = np.linalg.norm(d_vec, axis=1)
        return self.b[p_idx] - norm

    def predict_proba(self, s_idx, p_idx, o_idx):
        return 1 / (1 + np.exp(-self.score(s_idx, p_idx, o_idx)))
