import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PCGNN_ROOT = ROOT / "paper_aaai27" / "baseline_sources" / "PCGNN_recbole_drive" / "RecBole-master"
if str(PCGNN_ROOT) not in sys.path:
    sys.path.insert(0, str(PCGNN_ROOT))

from recbole.model.sequential_recommender.kg_model import kg_model  # noqa: E402


def numpy_session_graph_reference(item_seq: torch.Tensor):
    rows = item_seq.detach().cpu().numpy()
    max_n_node = rows.shape[1]
    aliases, matrices, items, positions = [], [], [], []
    mask = item_seq.gt(0)

    for u_input in rows:
        true_len = len(np.unique(u_input))
        position_index = [i for i in range(max_n_node)]
        position_index[0:true_len] = position_index[true_len - 1 :: -1]
        positions.append(position_index)

        node = np.unique(u_input)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1

        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        matrices.append(np.concatenate([u_A_in, u_A_out]).transpose())
        aliases.append([np.where(node == i)[0][0] for i in u_input])

    return (
        torch.as_tensor(aliases, dtype=torch.long),
        torch.as_tensor(np.asarray(matrices, dtype=np.float32)),
        torch.as_tensor(items, dtype=torch.long),
        mask,
        torch.as_tensor(positions, dtype=torch.long),
    )


class PCGNNSessionGraphTests(unittest.TestCase):
    def test_fast_session_graph_matches_numpy_reference(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        item_seq = torch.tensor(
            [
                [5, 7, 5, 9, 0, 0],
                [4, 4, 4, 0, 0, 0],
                [8, 0, 3, 0, 0, 0],
                [2, 3, 4, 5, 6, 7],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.long,
            device=device,
        )
        model = kg_model.__new__(kg_model)

        expected = numpy_session_graph_reference(item_seq)
        actual = model._build_session_graph_fast(item_seq)

        for expected_tensor, actual_tensor in zip(expected, actual):
            self.assertEqual(actual_tensor.device.type, device.type)
            torch.testing.assert_close(actual_tensor.cpu(), expected_tensor.cpu())


if __name__ == "__main__":
    unittest.main()
