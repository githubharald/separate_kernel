import numpy as np

from separate_kernel import separate_kernel


def all_close(a, b):
    return all(np.isclose(a.flatten(), b.flatten()))


def test_box_filter():
    M = np.ones([5, 5])
    res = separate_kernel(M, symmetric_kernel=True)
    assert all_close(res.col_vec, np.ones(5))
    assert all_close(res.row_vec, np.ones(5))


def test_box_filter_no_constraint():
    M = np.ones([5, 5])
    res = separate_kernel(M, symmetric_kernel=False)
    assert all_close(res.col_vec @ res.row_vec, M)


def test_rect_box_filter():
    M = np.ones([2, 3])
    res = separate_kernel(M, symmetric_kernel=False)
    assert all_close(res.col_vec @ res.row_vec, M)


def test_sobel_filter():
    M = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    res = separate_kernel(M, symmetric_kernel=False)
    assert all_close(res.col_vec @ res.row_vec, M)
