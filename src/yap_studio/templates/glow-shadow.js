function cardGlowShadow(color, level) {
  const rgb = [1, 3, 5].map(i => parseInt(color.slice(i, i + 2), 16)).join(",");
  return "0.02em 0 0 rgba(0,0,0,1),-0.02em 0 0 rgba(0,0,0,1)," +
    "0 0.02em 0 rgba(0,0,0,1),0 -0.02em 0 rgba(0,0,0,1)," +
    "0 0 0.05em rgba(0,0,0,0.85),0 0.03em 0.06em rgba(0,0,0,0.75)," +
    [0.06, 0.18, 0.4].map((blur, i) =>
      `0 0 ${blur}em rgba(${rgb},${level * [1, 0.75, 0.5][i]})`
    ).join(",");
}
