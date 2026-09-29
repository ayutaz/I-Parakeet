from iparakeet.analysis.kernels import kernel_report_markdown, softmax_errors, swish_kernel_errors


def test_swish_integer_kernels_follow_table3_ranking():
    rows = swish_kernel_errors(in_bits=8, alpha=8.0)
    by = {r["spec"]: r for r in rows}
    real = [by[s]["approx_max_err"] for s in ("poly:linf_swish", "poly:l2_swish", "poly:linf_tanh", "poly:l2_tanh", "hardswish")]
    assert real == sorted(real)
    assert abs(by["poly:linf_swish"]["approx_max_err"] - 0.039) < 5e-4
    assert by["poly:linf_swish"]["int_max_err"] <= 0.039 + 1.5 * by["poly:linf_swish"]["out_step"]
    assert by["lut"]["int_max_err"] <= 1.5 * by["lut"]["out_step"]


def test_log2_range_reduction_is_more_accurate_on_coarse_score_grids():
    rows = {(r["alpha"], r["range_reduction"]): r["max_abs_err"] for r in softmax_errors(alphas=(8.0, 32.0), bits=8)}
    assert rows[(32.0, "log2")] < rows[(32.0, "ibert")]
    assert rows[(8.0, "log2")] <= 2.5 / 127


def test_markdown_report_lists_both_tables():
    md = kernel_report_markdown(swish_kernel_errors(8, 8.0), softmax_errors((8.0,), 8))
    assert "poly:linf_swish" in md and "range reduction" in md
