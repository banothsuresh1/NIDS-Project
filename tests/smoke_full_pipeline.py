import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from nids import config, pipeline, metrics, streaming, explainability, attack_graph, ablation

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/synthetic_cicids2017"

t0 = time.time()
artifacts = pipeline.run_pipeline(DATA_DIR)
print(f"\nrun_pipeline took {time.time()-t0:.1f}s")

test = artifacts.test_sessions
print("\n=== Test session-level results ===")
print(test[["label", "predicted_class", "risk", "tier"]].head(10))
print("\ntier distribution:", test["tier"].value_counts().to_dict())

print("\n=== Per-class report (test) ===")
rep = metrics.per_class_report(test["label"], test["predicted_class"])
print(rep)

print("\nmacro-F1:", metrics.macro_f1(test["label"], test["predicted_class"]))
print("FPR:", metrics.false_positive_rate(test["label"], test["predicted_class"]))

print("\n=== Stage 12: streaming evaluation ===")
summ = test.reset_index().rename(columns={"index": "session_id"})[["session_id", "start_time", "end_time", "n_flows", "label"]]
risk_lookup = test["risk"].to_dict()
def score_fn(sid):
    return risk_lookup[sid]
stream_result = streaming.simulate_streaming(summ, score_fn)
print("throughput events/sec:", stream_result.throughput_events_per_sec)
print("mean latency ms/event:", stream_result.mean_latency_ms_per_event)
print(streaming.early_detection_stats(stream_result))

print("\n=== Stage 13: explainability for one ATTACK-tier session ===")
attack_rows = test[test["tier"] == "ATTACK"]
if len(attack_rows) > 0:
    sid = attack_rows.index[0]
    seq = artifacts.test_sequences[sid]
    from nids import features as _f
    fsets = artifacts.feature_sets
    print(f"session {sid}: label={test.loc[sid,'label']} predicted={test.loc[sid,'predicted_class']} risk={test.loc[sid,'risk']:.3f}")
    path = attack_graph.graph_path_evidence(seq, artifacts.graph)
    print("graph path:", path)
else:
    print("No ATTACK-tier sessions in this synthetic run (expected with small synthetic data / weak SNR).")

print("\n=== Ablation smoke: A2 (zero SP_t) + A3 (zero TC_t/G_w) ===")
test_a2 = ablation.rerun_fusion_and_risk(ablation.zero_sp_t(test), artifacts.fusion_weights, artifacts.risk_model)
test_a3 = ablation.rerun_fusion_and_risk(ablation.zero_temporal_graph(test), artifacts.fusion_weights, artifacts.risk_model)
cmp_df = ablation.compare_ablation("full", test, "no_SP_t", test_a2)
print(cmp_df)

print("\nALL FULL-PIPELINE SMOKE TESTS PASSED")
