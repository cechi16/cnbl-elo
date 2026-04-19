import https from "https";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  const token = process.env.GH_DISPATCH_TOKEN;
  if (!token) return res.status(500).json({ error: "GH_DISPATCH_TOKEN not configured" });

  const body = JSON.stringify({ event_type: "scrape" });

  await new Promise((resolve) => {
    const r = https.request(
      {
        hostname: "api.github.com",
        path: "/repos/cechi16/cnbl-elo/dispatches",
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: "application/vnd.github.v3+json",
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
          "User-Agent": "cnbl-elo",
        },
      },
      (resp) => {
        if (resp.statusCode === 204) res.status(200).json({ ok: true });
        else res.status(resp.statusCode).json({ error: `GitHub: ${resp.statusCode}` });
        resolve();
      }
    );
    r.on("error", (e) => { res.status(500).json({ error: e.message }); resolve(); });
    r.write(body);
    r.end();
  });
}
