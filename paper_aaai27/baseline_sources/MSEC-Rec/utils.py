import pandas as pd
import numpy as np
import torch
import dgl
import math


def generate_data(n_user, m_item, allPos, batch_size):
    users = np.arange(n_user)
    np.random.shuffle(users)
    num_neg = 1
    for i in range(0, n_user, batch_size):
        pos_items, neg_items = [], []

        for user in users[i:i + batch_size]:

            pos_item = np.random.choice(allPos[user])
            pos_items.append(pos_item)
            user_negs = []
            for _ in range(num_neg):
                neg_item = np.random.randint(m_item)
                while neg_item in allPos[user]:
                    neg_item = np.random.randint(m_item)
                user_negs.append(neg_item)
            neg_items.append(user_negs)
        yield (
            torch.tensor(users[i:i + batch_size], dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long)
        )


def mrr(n_user, m_item, allPos, testPos, model, device, k=10, batch_size=256):
    model.eval()
    mrrs = []
    for user_id in range(0, n_user, batch_size):
        users, items = [], []
        for i in range(batch_size):
            if user_id + i >= n_user:
                break
            users.extend([user_id + i] * k)
            pos_item = testPos[user_id + i]
            batch_items = [pos_item]
            for _ in range(k - 1):
                t = np.random.randint(m_item)
                while t in allPos[user_id + i] or t == pos_item:
                    t = np.random.randint(m_item)
                batch_items.append(t)
            items.extend(batch_items)
        users = torch.tensor(users).long().to(device)
        items = torch.tensor(items).long().to(device)
        with torch.no_grad():
            scores = model.bprmf.forward(users, items).cpu().numpy()
        for i in range(0, len(scores), k):
            user_scores = scores[i:i + k]
            item_score = list(zip(items[i:i + k].cpu().numpy(), user_scores))
            item_score.sort(key=lambda x: x[1], reverse=True)
            ranked_items = [x[0] for x in item_score]
            pos_item = testPos[users[i].item()]
            if pos_item in ranked_items:
                rank = ranked_items.index(pos_item) + 1
                mrrs.append(1 / rank)
            else:
                mrrs.append(0)
    return np.mean(mrrs)

def getNDCG(ranklist, gtItem):

    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0

def ndcg(n_user, m_item, allPos, testPos, model, device, k=10):

    model.eval()
    ndcgs = []

    for user_id in range(n_user):
        users = [user_id] * 100
        items = [testPos[user_id]]
        for i in range(99):
            t = np.random.randint(m_item)
            while t in allPos[user_id] or t == items[0]:
                t = np.random.randint(m_item)
            items.append(t)

        users = torch.Tensor(users).long().to(device)
        items = torch.Tensor(items).long().to(device)

        with torch.no_grad():
            scores = model.bprmf.forward(users, items).cpu().numpy()


        item_score = list(zip(items.cpu().numpy(), scores))
        item_score.sort(key=lambda x: x[1], reverse=True)


        top_k_items = [x[0] for x in item_score[:k]]


        ndcg_value = getNDCG(top_k_items, testPos[user_id])
        ndcgs.append(ndcg_value)

    return np.mean(ndcgs)

def hr(n_user, m_item, allPos, testPos, model, device, k=10):
    model.eval()
    hr = []
    for user_id in range(n_user):
        users = [user_id] * 100
        items = [testPos[user_id]]
        for i in range(99):
            t = np.random.randint(m_item)
            while t in allPos[user_id] or t == items[0]:
                t = np.random.randint(m_item)
            items.append(t)
        users = torch.Tensor(users).long().to(device)
        items = torch.Tensor(items).long().to(device)
        with torch.no_grad():
            scores = model.bprmf.forward(users, items).cpu().numpy()
        item_score = list(zip(items.cpu().numpy(), scores))
        item_score.sort(key=lambda x: x[1], reverse=True)
        top_k_items = [x[0] for x in item_score[:k]]
        if testPos[user_id] in top_k_items:
            hr.append(1)
        else:
            hr.append(0)
    return np.mean(hr)



