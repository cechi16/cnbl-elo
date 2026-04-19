module.exports = async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  const token = process.env.GH_DISPATCH_TOKEN;
  if (!token) return res.status(500).json({ error: "GH_DISPATCH_TOKEN not configured" });

  const r = await fetch("https://api.github.com/repos/cechi16/cnbl-elo/dispatches", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github.v3+json",
      "Content-Type": "application/json",
      "User-Agent": "cnbl-elo",
    },
    body: JSON.stringify({ event_type: "scrape" }),
  });

  if (r.status === 204) return res.status(200).json({ ok: true });
  const text = await r.text();
  return res.status(r.status).json({ error: text });
};
