import pytest

from humming.tune._worker import indexed_input_rows


def test_indexed_input_rows_w13_is_per_token():
    # shape_m is routed rows; w13 consumes tokens = shape_m // top_k.
    activation_rows, routing_tokens = indexed_input_rows(
        shape_m=2048, top_k=8, is_moe_down=False
    )
    assert routing_tokens == 256
    assert activation_rows == 256
    # routed rows produced by the generator == shape_m.
    assert routing_tokens * 8 == 2048


def test_indexed_input_rows_w2_is_routed_rows():
    activation_rows, routing_tokens = indexed_input_rows(
        shape_m=2048, top_k=8, is_moe_down=True
    )
    assert routing_tokens == 256
    assert activation_rows == 2048


def test_indexed_input_rows_requires_divisible_shape_m():
    with pytest.raises(ValueError, match="divisible by top_k"):
        indexed_input_rows(shape_m=2049, top_k=8, is_moe_down=False)
