"""CPU-only checks of released public-agent and paired-intervention artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_public_agent_light_v4 import long_estimates, aggregate_metrics

ROOT = Path(__file__).resolve().parents[1]

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
    result={'passed':True,'public_agent_episodes':len(episodes),'public_agent_metric_rows':len(actual),
            'paired_intervention_trajectories':trajectories,'table_3_rounding_checks':12}
    out=ROOT/'build/paper_release_verification.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
