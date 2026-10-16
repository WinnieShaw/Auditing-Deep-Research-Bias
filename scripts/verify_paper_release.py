"""CPU-only checks of released public-agent and paired-intervention artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_public_agent_light_v4 import long_estimates, aggregate_metrics

ROOT = Path(__file__).resolve().parents[1]


def verify_paper_figures():
    figures = ROOT / 'figures'
    required = [
        figures / 'overview.pdf',
        figures / 'overview_editable.pptx',
        figures / 'fig_search_bias_and_stability.pdf',
        figures / 'fig_search_bias_and_stability.svg',
        figures / 'fig_budget_efficiency.pdf',
        figures / 'fig_budget_efficiency.svg',
    ]
    for path in required:
        assert path.is_file() and path.stat().st_size > 0, path

    ranking = pd.read_csv(figures / 'source_data/fig_search_bias_and_stability.csv')
    assert len(ranking) == 6
    expected_ranking = {
        ('a', 'controlled', 'uncorrected'): 0.7478,
        ('a', 'controlled', 'causal_correction'): 0.0014,
        ('b', 'Qwen2.5-32B', 'uncorrected'): 0.0415,
        ('b', 'Qwen2.5-32B', 'CESS'): 0.0185,
        ('b', 'OLMo3-7B', 'uncorrected'): 0.0391,
        ('b', 'OLMo3-7B', 'CESS'): 0.0076,
    }
    actual_ranking = {
        (row.panel, row.setting, row.method): float(row.conclusion_gap)
        for row in ranking.itertuples(index=False)
    }
    assert actual_ranking == expected_ranking

    budget = pd.read_csv(figures / 'source_data/fig_budget_efficiency.csv')
    assert len(budget) == 16
    assert set(budget['model']) == {'Qwen2.5-32B', 'OLMo3-7B'}
    assert set(budget['search_rounds']) == {1, 2, 3, 4}
    assert set(budget['method']) == {'uncorrected', 'Task-wise CESS'}
    expected_gains = {
        'Qwen2.5-32B': np.array([41.5, 23.4, 15.8, 9.1]),
        'OLMo3-7B': np.array([40.7, 27.0, 18.1, 15.8]),
    }
    for model, expected in expected_gains.items():
        rows = budget[budget.model == model]
        opened = rows[rows.method == 'uncorrected'].sort_values('search_rounds').mae.to_numpy()
        cess = rows[rows.method == 'Task-wise CESS'].sort_values('search_rounds').mae.to_numpy()
        assert np.allclose(np.round(100 * (opened - cess) / opened, 1), expected)
    return len(required), len(ranking) + len(budget)

def main():
    base = ROOT / 'outputs/public_agent_light_v4'
    episodes = [json.loads(s) for s in (base/'test/episodes.jsonl').read_text().splitlines()]
    config = json.loads((base/'frozen_test_config.json').read_text())
    assert len(episodes) == 216
    assert len({e['review_id'] for e in episodes}) == 36
    long, diagnostics = long_estimates(episodes, config)
    actual = aggregate_metrics(long).sort_values(['horizon','method']).reset_index(drop=True)
    reference = pd.read_csv(base/'analysis_final/method_metrics.csv').sort_values(['horizon','method']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, reference, check_dtype=False, atol=1e-12, rtol=1e-10)
    for e in episodes:
        assert not e.get('error') and len(e['searches']) == 4
        for search in e['searches']:
            assert 0 < search['selected_probability'] <= 1
    expected = {'Opened Mean':(.5727,.5849,.4907), 'Outcome Regression':(.2404,0,.3333),
                'Sequential DR':(.3053,.1492,.3287), 'Full CESS':(.2287,.0746,.3102)}
    for method, values in expected.items():
        row = actual[(actual.horizon==2)&(actual.method==method)].iloc[0]
        assert np.allclose(row[['mae','ranking_sensitivity','wrong_direction_rate']].to_numpy(float),values,atol=0.00005,rtol=0)
    trajectories = 0
    for model in ['qwen2_5_32b','olmo3_7b']:
        p=ROOT/'outputs/interpretability_v1_true_agent_ms2'/model/'cells.jsonl'
        cells=pd.read_json(p,lines=True)
        assert len(cells)==2400 and cells.review_id.nunique()==200
        trajectories += len(cells)
    figure_files, figure_source_rows = verify_paper_figures()
    result={'passed':True,'public_agent_episodes':len(episodes),'public_agent_metric_rows':len(actual),
            'paired_intervention_trajectories':trajectories,'table_3_rounding_checks':12,
            'paper_figure_files':figure_files,'paper_figure_source_rows':figure_source_rows}
    out=ROOT/'build/paper_release_verification.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
