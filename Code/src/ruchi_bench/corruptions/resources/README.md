# Vendored visual-confusion resource

`visual_confusions.json` is a compact snapshot derived from the `similar_font_chars`
field of [Macielyoung/Confused_Chinese](https://github.com/Macielyoung/Confused_Chinese),
an Apache-2.0 project. It retains the top five Chinese candidates for each source
character and removes self-loops/non-CJK candidates. The generator keeps the source
ordering and similarity ranking; it does not infer new pairs.

The full upstream resource can be refreshed when needed. The benchmark code only
reads this deterministic snapshot at runtime, so experiments do not depend on a
network connection or a model-generated confusion table.

