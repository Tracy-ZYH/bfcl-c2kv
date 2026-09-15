"""Read-only post-run validator; no HTTP/model/accelerator imports."""
import argparse
import json
from pathlib import Path


def validate_episode(row):
    events = row.get("history_kv_lifecycle", [])
    opened = [e for e in events if e.get("event") == "session_open"]
    closed = [e for e in events if e.get("event") == "session_close"]
    assert len(opened) == len(closed) == 1, f"{row['id']}: session not opened/closed exactly once"
    sid = opened[0]['session_id']
    assert closed[0]['session_id'] == sid and closed[0].get('closed'), 'session close mismatch'
    requests = [e for e in events if e.get('event') == 'session_prompt_saved']
    assert requests, f"{row['id']}: no final resident cache telemetry"
    previous = None
    for e in requests:
        assert e['session_id'] == sid and e['episode_id'] == row['id'], 'session/episode changed'
        assert e['persistent_session_enabled'] and e['history_kv_backend'] == 'physical_eviction'
        assert e['full_history_reprefill_performed'] is False, 'unexpected full reprefill'
        assert e['resident_tokens_after_append'] == e['resident_tokens_before_append'] + e['new_turn_tokens']
        assert e['resident_tokens_after_eviction'] == e['resident_tokens_after_append'] - e['evicted_tokens_this_turn']
        assert e['resident_tokens_after_eviction'] == e['resident_position_summary']['count']
        if previous:
            assert e['previous_resident_position_summary'] == previous, 'previous retained cache not reused'
        previous = e['resident_position_summary']
    return sid, len(requests), len({e['turn_id'] for e in requests})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--expected-examples', type=int, required=True)
    p.add_argument('--methods', default='streamingllm,h2o,snapkv_persistent,pyramidkv')
    a = p.parse_args()
    released = set()
    for log in a.run_root.glob('server_*.log'):
        for line in log.open(errors='replace'):
            if 'HISTORY_KV_SESSION_CLOSED ' in line:
                released.add(json.loads(line.split('HISTORY_KV_SESSION_CLOSED ', 1)[1])['session_id'])
    ids = None
    for m in a.methods.split(','):
        if m not in {'streamingllm', 'h2o', 'snapkv', 'snapkv_persistent', 'pyramidkv'}:
            continue
        path = a.run_root/m/'logs/details.jsonl'
        rows = [json.loads(l) for l in path.open() if l.strip()]
        current_ids = [r['id'] for r in rows]
        assert len(current_ids) == len(set(current_ids)) == a.expected_examples, f'{m}: incomplete episodes'
        assert ids is None or ids == current_ids, 'baseline episode IDs/order differ'
        ids = current_ids
        requests = multi_turn = 0
        for row in rows:
            sid, n, turns = validate_episode(row)
            assert sid in released, f'{sid}: no server-side resource release record'
            requests += n
            multi_turn += turns > 1
        print(f'PASS {m}: episodes={len(rows)} requests={requests} multi_turn_episodes={multi_turn}; session continuity and release verified')


if __name__ == '__main__':
    main()
