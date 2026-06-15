import json, sys
from pathlib import Path

def main():
    from graphify.detect import detect
    from graphify.extract import collect_files, extract
    from graphify.build import build_from_json
    from graphify.cluster import cluster, score_all
    from graphify.analyze import god_nodes, surprising_connections, suggest_questions
    from graphify.report import generate
    from graphify.export import to_json

    Path("graphify-out").mkdir(exist_ok=True)

    print("Running detect...")
    result = detect(Path('.'))
    with open('graphify-out/.graphify_detect.json', 'w', encoding='utf-8') as f:
        json.dump(result, f)

    code_files = []
    for f in result.get('files', {}).get('code', []):
        p = Path(f)
        code_files.extend(collect_files(p) if p.is_dir() else [p])

    print("Running extract...")
    if code_files:
        ast_res = extract(code_files, cache_root=Path('.'))
    else:
        ast_res = {'nodes':[],'edges':[],'input_tokens':0,'output_tokens':0}

    merged = {
        'nodes': ast_res['nodes'],
        'edges': ast_res['edges'],
        'hyperedges': [],
        'input_tokens': 0,
        'output_tokens': 0,
    }

    print(f"Extracted {len(merged['nodes'])} nodes.")

    print("Building and clustering graph...")
    G = build_from_json(merged)
    if G.number_of_nodes() == 0:
        print('ERROR: Graph is empty')
        sys.exit(1)

    communities = cluster(G)
    cohesion = score_all(G, communities)
    tokens = {'input': 0, 'output': 0}
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    labels = {cid: 'Community ' + str(cid) for cid in communities}
    questions = suggest_questions(G, communities, labels)

    print("Generating report...")
    report = generate(G, communities, cohesion, labels, gods, surprises, result, tokens, '.', suggested_questions=questions)
    Path('graphify-out/GRAPH_REPORT.md').write_text(report, encoding="utf-8")
    to_json(G, communities, 'graphify-out/graph.json')

    import subprocess
    subprocess.run([sys.executable, "-m", "graphify", "export", "html"], check=False)
    print("SUCCESS")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
