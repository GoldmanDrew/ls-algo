/**
 * Investor login for the ls-algo risk dashboard (etf-dashboard pattern).
 *
 * Passwords are verified client-side with PBKDF2-HMAC-SHA256 against
 * site/data/investors.json (hashes only). Session lives in sessionStorage.
 */

(function () {
  const INVESTORS_URL = "./data/investors.json";
  const AUTH_STORAGE_KEY = "ls_risk_dash_session_v1";
  const AUTH_SESSION_MS = 7 * 86400000;

  function timingSafeEqualBytes(a, b) {
    if (!(a instanceof Uint8Array) || !(b instanceof Uint8Array) || a.length !== b.length) {
      return false;
    }
    let x = 0;
    for (let i = 0; i < a.length; i++) x |= a[i] ^ b[i];
    return x === 0;
  }

  function readAuthSession(validUserIds) {
    try {
      const raw = sessionStorage.getItem(AUTH_STORAGE_KEY);
      if (!raw) return null;
      const s = JSON.parse(raw);
      if (!s || typeof s.uid !== "string" || typeof s.exp !== "number") return null;
      if (Date.now() > s.exp) {
        sessionStorage.removeItem(AUTH_STORAGE_KEY);
        return null;
      }
      const uid = s.uid.toLowerCase();
      if (!validUserIds.has(uid)) {
        sessionStorage.removeItem(AUTH_STORAGE_KEY);
        return null;
      }
      return uid;
    } catch {
      return null;
    }
  }

  function writeAuthSession(uid) {
    sessionStorage.setItem(
      AUTH_STORAGE_KEY,
      JSON.stringify({ uid: uid.toLowerCase(), exp: Date.now() + AUTH_SESSION_MS })
    );
  }

  function clearAuthSession() {
    sessionStorage.removeItem(AUTH_STORAGE_KEY);
  }

  async function verifyInvestorPassword(userId, password, users) {
    const uid = userId.trim().toLowerCase();
    const u = users.find((x) => String(x.id || "").toLowerCase() === uid);
    if (!u || !u.salt_b64 || !u.hash_b64) return false;
    const iterations = Number(u.iterations) > 0 ? Number(u.iterations) : 250000;
    let salt;
    let expected;
    try {
      salt = Uint8Array.from(atob(String(u.salt_b64)), (c) => c.charCodeAt(0));
      expected = Uint8Array.from(atob(String(u.hash_b64)), (c) => c.charCodeAt(0));
    } catch {
      return false;
    }
    if (expected.length !== 32) return false;
    const enc = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
      "raw",
      enc.encode(password),
      "PBKDF2",
      false,
      ["deriveBits"]
    );
    const bits = await crypto.subtle.deriveBits(
      { name: "PBKDF2", salt, iterations, hash: "SHA-256" },
      keyMaterial,
      256
    );
    return timingSafeEqualBytes(new Uint8Array(bits), expected);
  }

  function parseInvestorUsers(doc) {
    if (!doc || !Array.isArray(doc.users)) return [];
    return doc.users.filter((u) => u && u.id && u.salt_b64 && u.hash_b64);
  }

  async function loadInvestors() {
    const bust = `?t=${Date.now()}`;
    try {
      const res = await fetch(INVESTORS_URL + bust, { cache: "no-store" });
      if (!res.ok) {
        return { users: [], authEnabled: false };
      }
      const doc = await res.json();
      const users = parseInvestorUsers(doc);
      return { users, authEnabled: users.length > 0 };
    } catch {
      return { users: [], authEnabled: false };
    }
  }

  async function verifyLogin(userId, password, users) {
    return verifyInvestorPassword(userId, password, users);
  }

  function getStoredSession(validUserIds) {
    return readAuthSession(validUserIds);
  }

  window.LSAuth = {
    INVESTORS_URL,
    loadInvestors,
    verifyLogin,
    getStoredSession,
    writeAuthSession,
    clearAuthSession,
  };
})();
