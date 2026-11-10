from RL.grpo.groups import iter_group_batches


class Group:
    def __init__(self, size):
        self.group_size = size


def test_group_windows_keep_groups_intact_and_retain_tail():
    groups = [Group(2), Group(3), Group(4)]
    windows = list(iter_group_batches(groups, 2))
    assert [[g.group_size for g in window] for window in windows] == [[2, 3], [4]]
