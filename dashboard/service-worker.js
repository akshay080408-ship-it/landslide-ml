// LandSense ML — Service Worker
// WHY: the offline queue (in report.html) already handles
// submitting when the API is unreachable. This adds the
// other half: making the PAGE ITSELF load with zero network
// at all — genuinely useful for a field officer in a
// low/no-connectivity area who needs to open the form before
// they even have a connection to submit through.
//
// HONEST SCOPE: this caches report.html's own static assets
// (this page, the manifest, the icons). It does NOT cache
// live API data (nodes/alerts/reports) — that's still
// something the map dashboard needs a real connection for.
// This is "the report form works offline," not "the whole
// platform works offline."

const CACHE_NAME = 'landsense-report-v1';
const ASSETS_TO_CACHE = [
  './report.html',
  './manifest.json',
  './icon-192.png',
  './icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS_TO_CACHE))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // Only intercept requests for the cached static assets —
  // API calls (fetch to 127.0.0.1:5000/*) pass straight
  // through untouched, so the offline-queue logic in
  // report.html still works exactly as it did before.
  const url = new URL(event.request.url);
  const isStaticAsset = ASSETS_TO_CACHE.some((asset) => url.pathname.endsWith(asset.replace('./', '')));

  if (isStaticAsset) {
    event.respondWith(
      caches.match(event.request).then((cached) => cached || fetch(event.request))
    );
  }
});
