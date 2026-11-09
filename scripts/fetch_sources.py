"""按锁定版本下载供阅读的上游源码，保留在被 Git 忽略的 external/。"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import tarfile
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def source_files(archive_path):
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            records = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                content = archive.read(info)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    target = PurePosixPath(info.filename).parent / content.decode("utf-8")
                    if target.is_absolute() or ".." in target.parts:
                        raise ValueError("符号链接必须指向压缩包内同级文件")
                    linked_info = archive.getinfo(target.as_posix())
                    if (linked_info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("不支持连续符号链接")
                    # 将 SB3 的 .dockerignore → .gitignore 保存为常规文件副本。
                    content = archive.read(linked_info)
                records.append((info.filename, content))
    else:
        with tarfile.open(archive_path) as archive:
            members = [member for member in archive.getmembers() if not member.isdir()]
            if not all(member.isfile() for member in members):
                raise ValueError("源码压缩包包含非常规文件")
            records = [(member.name, archive.extractfile(member).read()) for member in members]
    files = []
    for name, content in records:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or any(":" in part or "\\" in part for part in path.parts):
            raise ValueError("压缩包包含不安全路径")
        relative = PurePosixPath(*path.parts[1:])
        if not relative.parts:
            raise ValueError("源码条目缺少根目录")
        files.append((relative.as_posix(), content))
    return files


def tree_digest(files):
    digest = hashlib.sha256()
    for relative, content in sorted(files, key=lambda item: item[0]):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def fetch(source):
    external = (ROOT / "external").resolve()
    destination = (ROOT / source["directory"]).resolve()
    destination.relative_to(external)
    marker = destination / ".source-info.json"
    if destination.exists():
        if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))["revision"] == source["revision"]:
            print("源码已就绪", source["id"], destination)
            return
        raise FileExistsError(f"目录已有内容，请先核对版本 {destination}")
    cache = ROOT / ".wheelhouse/source_archives"
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / source["archive"]
    if not archive_path.exists():
        partial = archive_path.with_name(archive_path.name + ".part")
        url = source["url"]
        started = time.monotonic()
        print("正在下载", source["id"], source["revision"], flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "rl-study-setup"})
        with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as output:
            while True:
                chunk = response.read1(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                if output.tell() > 256 * 1024 * 1024 or time.monotonic() - started > 600:
                    raise RuntimeError("源码下载超过体积或时间上限")
        partial.replace(archive_path)
    external.mkdir(exist_ok=True)
    if source["kind"] == "github":
        with zipfile.ZipFile(archive_path) as archive:
            revision = archive.comment.decode("ascii").strip()
        if revision != source["revision"]:
            raise ValueError("缓存压缩包的提交号与清单不一致")
    elif hashlib.sha256(archive_path.read_bytes()).hexdigest() != source["archive_sha256"]:
        raise ValueError("发行包 SHA-256 与官方记录不一致")
    files = source_files(archive_path)
    if tree_digest(files) != source["tree_sha256"]:
        raise ValueError("源码文件内容与锁定版本不一致")
    with tempfile.TemporaryDirectory(dir=external, prefix=".fetch-") as temporary:
        staging = Path(temporary).resolve() / source["id"]
        staging.relative_to(external)
        staging.mkdir()
        seen = set()
        for relative, content in files:
            target = (staging / relative).resolve()
            target.relative_to(staging)
            key = str(target).casefold()
            if key in seen:
                raise ValueError("源码压缩包包含大小写冲突的路径")
            seen.add(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (staging / ".source-info.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
        # 只将本次创建的临时目录移入已验证的 external 子目录。
        destination.relative_to(external)
        staging.rename(destination)
    print("源码已就绪", source["id"], len(files), "个文件", destination)


def main():
    sources = json.loads((ROOT / "docs/source_manifest.json").read_text(encoding="utf-8"))["sources"]
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", choices=["all", *[source["id"] for source in sources]], default="all")
    args = cli.parse_args()
    for source in sources:
        if args.source in ("all", source["id"]):
            fetch(source)


if __name__ == "__main__":
    main()
