from iparakeet.sim.recipe import PAPER_TABLE2, PAPER_TABLE3
from iparakeet.sim.report import check_table2_order, check_table3_order, table2_markdown, table3_markdown


def _paper_results():
    results = {name: {"test-clean": v[0], "test-other": v[1], "commonvoice-test": v[2]} for name, v in PAPER_TABLE2.items()}
    for row in PAPER_TABLE3:
        results.setdefault(row.recipe, {})["test-other"] = row.wer_test_other
    return results


def test_paper_numbers_pass_their_own_order_checks():
    results = _paper_results()
    assert check_table2_order(results, fp32={"test-clean": 1.87, "test-other": 3.76, "commonvoice-test": 10.55}) == {
        "test-clean": True, "test-other": True, "commonvoice-test": True,
    }
    assert check_table3_order(results) == {"swish": True, "bn": True, "calib": True}


def test_order_checks_detect_swaps_but_tolerate_near_ties():
    results = _paper_results()
    results["swish_hardswish"]["test-other"] = 5.40  # now better than L2/Swish -> order broken
    results["bn_per_channel_int16"]["test-other"] = 5.35  # 0.03 worse than per-tensor: within the tie rule
    checks = check_table3_order(results)
    assert checks["swish"] is False
    assert checks["bn"] is True
    results["naive_int8"]["test-other"] = 8.0
    assert check_table2_order(results, fp32={"test-other": 3.76})["test-other"] is False


def test_markdown_tables_show_reproduced_and_paper_values():
    results = _paper_results()
    results["iparakeet"]["test-other"] = 5.50
    md2 = table2_markdown(results, fp32={"test-other": 3.80})
    assert "| iparakeet |" in md2 and "5.50" in md2 and "5.32" in md2 and "+0.18" in md2
    md3 = table3_markdown(results)
    assert md3.count("\n|") >= 12 and "Hard-Swish" in md3
    assert "n/a" in table3_markdown({})
