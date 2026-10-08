# (Copied from SetRetrieval 49e30bd methods/heterogeneous_diffusion/colab/hf_prune_checkpoints.py; only the default
#  token path changed. Runs on a machine outside the notebook, alongside a training run. Deletion cannot be undone.)
# Hugging Face のリポジトリから、置き換えられて参照されなくなったチェックポイント (latest.pt) を完全に消す。
#
# 理由: ブランチの履歴をまとめても (super_squash_history)、ブランチを消しても、LFS の実体は容量に数えられたまま残る。
#       無料枠の非公開の容量は 100GB で、12〜13GB のチェックポイントを 20 分ごとに置き換えると、すぐに埋まる。
#
# 消す対象 (すべて満たすもの):
#   - ファイル名が latest.pt
#   - どのブランチの先頭にも、その実体 (sha256) を指すファイルが無い
#   - 送ってから --min_age_minutes 分以上たっている (送信の途中のものを消さないため)
# 特徴のキャッシュ (cache-*) と、学習後に残す重みには触れない。
#
# 識別子の注意: 一覧 (lfs-files) の fileOid が実体の sha256 (ブランチのファイルの lfs.sha256 と同じ値)。
#   一覧の oid は、ポインタファイルの git の blob id であって sha256 ではない。照合と削除には fileOid を使う
#   (公式の huggingface_hub の permanently_delete_lfs_files も file_oid を送る)。
#   取り違えると「どのブランチも指していない」と誤って判定するので、照合が成り立つことを毎回確かめ、成り立たなければ何も消さない。
#
#   python hf_prune_checkpoints.py                 # 消す対象を表示するだけ
#   python hf_prune_checkpoints.py --execute       # 実際に消す (元に戻せない)
#   python hf_prune_checkpoints.py --execute --loop_minutes 5   # 5 分ごとに繰り返す
import argparse, datetime, json, os, time

import requests
from huggingface_hub import HfApi

NAME = "latest.pt"


def inspect(api, repo, headers):
    """(LFS の実体の一覧, ブランチの先頭が指している sha256 の集合)。"""
    referenced = set()
    for branch in api.list_repo_refs(repo).branches:
        for f in api.list_repo_tree(repo, revision=branch.name, recursive=True):
            lfs = getattr(f, "lfs", None)
            if lfs:
                referenced.add(lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256)
    rows, url = [], f"https://huggingface.co/api/models/{repo}/lfs-files"
    while url:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        rows += r.json()
        url = r.links.get("next", {}).get("url")
    return rows, referenced


def stale(rows, referenced, min_age_minutes):
    listed = {x["fileOid"] for x in rows}
    # ブランチの先頭が指す実体は、必ず一覧に載っているはず。載っていなければ識別子の対応が崩れているので、何も消さない
    if not referenced or not referenced <= listed or any(len(x["fileOid"]) != 64 for x in rows):
        raise RuntimeError(f"照合できない: ブランチが指す実体 {len(referenced)} 個のうち {len(referenced - listed)} 個が一覧に無い")
    now = datetime.datetime.now(datetime.timezone.utc)
    out = []
    for x in rows:
        pushed = datetime.datetime.fromisoformat(x["pushedAt"].replace("Z", "+00:00"))
        age = (now - pushed).total_seconds() / 60
        if x.get("filename") == NAME and x["fileOid"] not in referenced and age >= min_age_minutes:
            out.append(x)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="Dalby123/heterogeneous-diffusion")
    p.add_argument("--token_file", default=os.path.expanduser("~/.cache/huggingface/token"))
    p.add_argument("--min_age_minutes", default=8.0, type=float)
    p.add_argument("--execute", action="store_true", help="実際に消す。付けなければ表示だけ")
    p.add_argument("--loop_minutes", default=0.0, type=float, help="0 なら 1 回だけ")
    p.add_argument("--log", default="")
    args = p.parse_args()
    token = open(args.token_file).read().strip()
    api, headers = HfApi(token=token), {"Authorization": f"Bearer {token}"}

    def say(text):
        line = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%FT%TZ')} {text}"
        print(line, flush=True)
        if args.log:
            with open(args.log, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    while True:
        try:
            rows, referenced = inspect(api, args.repo, headers)
            targets = stale(rows, referenced, args.min_age_minutes)
            total = sum(x["size"] for x in rows) / 2**30
            if targets or not args.loop_minutes:
                say(f"実体 {len(rows)} 個、合計 {total:.1f}GB。消す対象 {len(targets)} 個、{sum(x['size'] for x in targets) / 2**30:.1f}GB")
            for x in targets:
                say(f"  {'消す' if args.execute else '対象'}: {x['pushedAt'][:19]} {x['size'] / 2**30:6.2f}GB {x['filename']} ref={x.get('ref')} sha256={x['fileOid'][:12]}")
            if args.execute and targets:
                r = requests.post(f"https://huggingface.co/api/models/{args.repo}/lfs-files/batch", headers=headers, timeout=120,
                                  json={"deletions": {"sha": [x["fileOid"] for x in targets], "rewriteHistory": False}})
                say(f"  削除の応答 {r.status_code} {r.text[:200]}")
        except Exception as e:  # 通信の失敗では止めない
            say(f"失敗: {e!r}"[:300])
        if not args.loop_minutes:
            break
        time.sleep(args.loop_minutes * 60)


if __name__ == "__main__":
    main()
