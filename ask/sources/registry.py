"""The set of sources loaded in one chat, and file-reference resolution."""
from __future__ import annotations

from pathlib import Path

from ask.sources.loaders import LoadError, Source, load_path


class SourceRegistry:
    def __init__(self) -> None:
        self.sources: dict[str, Source] = {}

    # ── add / remove ──
    def load(self, path: Path, hint: str | None = None) -> list[Source]:
        loaded = load_path(path, hint)
        out = []
        for src in loaded:
            # Reloading the same path replaces the old copy instead of duplicating it.
            for existing in list(self.sources.values()):
                if existing.path == src.path:
                    del self.sources[existing.name]
            src.name = self._unique_name(src.name)
            self.sources[src.name] = src
            out.append(src)
        return out

    def _unique_name(self, name: str) -> str:
        if name not in self.sources:
            return name
        i = 2
        while f"{name} ({i})" in self.sources:
            i += 1
        return f"{name} ({i})"

    def remove(self, name: str) -> bool:
        return self.sources.pop(name, None) is not None

    def restore(self, records: list[dict]) -> list[str]:
        """Reload sources saved with a chat. Returns error messages for any that failed."""
        errors = []
        for rec in records:
            try:
                srcs = load_path(Path(rec["path"]), rec.get("kind"))
                for s in srcs:
                    if len(srcs) == 1:
                        s.name = rec.get("name", s.name)
                    s.name = self._unique_name(s.name)
                    self.sources[s.name] = s
            except LoadError as exc:
                errors.append(f"{rec.get('name', rec.get('path'))}: {exc}")
            except Exception as exc:
                errors.append(f"{rec.get('name', rec.get('path'))}: {type(exc).__name__}: {exc}")
        return errors

    def records(self) -> list[dict]:
        return [s.record() for s in self.sources.values()]

    # ── lookup ──
    def of_kind(self, *kinds: str) -> list[Source]:
        return [s for s in self.sources.values() if s.kind in kinds]

    def get(self, name: str | None, *kinds: str) -> Source:
        """A source by name; with no name, the single (or most recent) one of `kinds`."""
        pool = self.of_kind(*kinds) if kinds else list(self.sources.values())
        if name:
            if name in self.sources and (not kinds or self.sources[name].kind in kinds):
                return self.sources[name]
            low = name.lower()
            matches = [s for s in pool if s.name.lower() == low or low in s.name.lower()
                       or Path(s.path).name.lower() == low]
            if len(matches) == 1:
                return matches[0]
            raise KeyError(f"No loaded {'/'.join(kinds) or ''} source called '{name}'. "
                           f"Loaded: {', '.join(s.name for s in pool) or 'none'}")
        if not pool:
            raise KeyError(f"No {' or '.join(kinds) or ''} source is loaded yet. "
                           "Ask the user for a path to load.")
        return max(pool, key=lambda s: s.loaded_at)

    def resolve_file(self, ref: str) -> tuple[Source, str]:
        """Resolve a file reference ('path', 'source:path' or 'source/path') to
        (source, relpath) across loaded repo/doc sources."""
        ref = ref.strip().strip("`'\"").replace("\\", "/")
        text_sources = self.of_kind("repo", "docs")
        # explicit "source:path" or "source::path"
        for sep in ("::", ":"):
            if sep in ref:
                head, tail = ref.split(sep, 1)
                if head in self.sources and self.sources[head].kind != "data":
                    return self._match_in(self.sources[head], tail)
        # "source/path"
        for s in text_sources:
            prefix = s.name + "/"
            if ref.startswith(prefix):
                try:
                    return self._match_in(s, ref[len(prefix):])
                except KeyError:
                    pass
        hits: list[tuple[Source, str]] = []
        for s in text_sources:
            try:
                hits.append(self._match_in(s, ref))
            except KeyError:
                continue
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise KeyError(f"File '{ref}' is not in any loaded repo/document source.")
        exact = [h for h in hits if h[1] == ref.lstrip("./")]
        if len(exact) == 1:
            return exact[0]
        raise KeyError(f"'{ref}' is ambiguous: " + ", ".join(f"{s.name}:{r}" for s, r in hits[:8])
                       + ". Prefix it with the source name, e.g. 'source:path'.")

    @staticmethod
    def _match_in(src: Source, rel: str) -> tuple[Source, str]:
        rel = rel.strip().lstrip("./").replace("\\", "/")
        if rel in src.files:
            return src, rel
        low = rel.lower()
        exact_ci = [f for f in src.files if f.lower() == low]
        if len(exact_ci) == 1:
            return src, exact_ci[0]
        ends = [f for f in src.files if f.lower().endswith("/" + low) or f.lower() == low]
        if len(ends) == 1:
            return src, ends[0]
        base = [f for f in src.files if Path(f).name.lower() == Path(low).name]
        if len(base) == 1 and "/" not in rel:
            return src, base[0]
        if len(ends) > 1 or len(base) > 1:
            cands = ends or base
            raise KeyError(f"'{rel}' matches several files in {src.name}: {', '.join(cands[:8])}")
        raise KeyError(f"'{rel}' not found in {src.name}")

    def describe(self) -> str:
        if not self.sources:
            return "Nothing is loaded yet."
        lines = []
        for s in self.sources.values():
            if s.kind == "data":
                cols = ", ".join(map(str, list(s.df.columns)[:40])) if s.df is not None else ""
                more = f" … (+{s.df.shape[1] - 40} more)" if s.df is not None and s.df.shape[1] > 40 else ""
                sheets = f"; sheets: {', '.join(map(str, s.sheets))}" if len(s.sheets) > 1 else ""
                lines.append(f"- [data] {s.name} — {s.summary()} — path {s.path}{sheets}\n  columns: {cols}{more}")
            else:
                lines.append(f"- [{s.kind}] {s.name} — {s.summary()} — path {s.path}")
        return "\n".join(lines)
