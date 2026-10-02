/* ══════════════════════════════════════════════
   Smart Campus AI — Shared JS (design-system.js)
   Cursor · Particles · Transitions · Parallax
══════════════════════════════════════════════ */

(function() {
  'use strict';

  /* ── Loader ── */
  window.addEventListener('load', function() {
    var loader = document.getElementById('loader');
    if (!loader) return;
    setTimeout(function() {
      loader.classList.add('hidden');
      setTimeout(function() { loader.remove(); }, 600);
    }, 1200);
  });

  /* ── Custom cursor ── */
  var cursor     = document.getElementById('cursor');
  var cursorRing = document.getElementById('cursor-ring');
  var mouseX = 0, mouseY = 0;
  var ringX  = 0, ringY  = 0;

  if (cursor && cursorRing) {
    document.addEventListener('mousemove', function(e) {
      mouseX = e.clientX; mouseY = e.clientY;
      cursor.style.transform = 'translate(' + mouseX + 'px, ' + mouseY + 'px) translate(-50%,-50%)';
    });
    (function animRing() {
      ringX += (mouseX - ringX) * 0.12;
      ringY += (mouseY - ringY) * 0.12;
      cursorRing.style.transform = 'translate(' + ringX + 'px, ' + ringY + 'px) translate(-50%,-50%)';
      requestAnimationFrame(animRing);
    })();
  }

  /* ── Parallax bg on mouse move ── */
  var bgScene = document.getElementById('bgScene');
  if (bgScene) {
    document.addEventListener('mousemove', function(e) {
      var dx = (e.clientX / window.innerWidth  - 0.5) * 18;
      var dy = (e.clientY / window.innerHeight - 0.5) * 12;
      bgScene.style.transform = 'translate(' + dx + 'px, ' + dy + 'px) scale(1.04)';
    });
  }

  /* ── Floating elements parallax ── */
  var floatEls = document.querySelectorAll('.float-el');
  if (floatEls.length) {
    document.addEventListener('mousemove', function(e) {
      var cx = window.innerWidth  / 2;
      var cy = window.innerHeight / 2;
      floatEls.forEach(function(el, i) {
        var factor = (i % 2 === 0 ? 0.012 : 0.008);
        var dx = (e.clientX - cx) * factor;
        var dy = (e.clientY - cy) * factor;
        el.style.marginLeft = dx + 'px';
        el.style.marginTop  = dy + 'px';
      });
    });
  }

  /* ── Card radial spotlight + 3D tilt ── */
  document.querySelectorAll('.sc-card-tilt, .nav-card').forEach(function(card) {
    card.addEventListener('mousemove', function(e) {
      var r  = card.getBoundingClientRect();
      var x  = ((e.clientX - r.left) / r.width  * 100).toFixed(1);
      var y  = ((e.clientY - r.top)  / r.height * 100).toFixed(1);
      card.style.setProperty('--mx', x + '%');
      card.style.setProperty('--my', y + '%');

      var cx = r.left + r.width  / 2;
      var cy = r.top  + r.height / 2;
      var rx = ((e.clientY - cy) / (r.height / 2)) * -6;
      var ry = ((e.clientX - cx) / (r.width  / 2)) *  6;
      card.style.transform = 'translateY(-6px) scale(1.02) perspective(600px) rotateX(' + rx + 'deg) rotateY(' + ry + 'deg)';
    });
    card.addEventListener('mouseleave', function() {
      card.style.transform = '';
    });
  });

  /* ── Page transition on data-dest links ── */
  document.querySelectorAll('[data-dest]').forEach(function(el) {
    el.addEventListener('click', function(e) {
      e.preventDefault();
      var dest    = el.dataset.dest;
      var overlay = document.getElementById('transition-overlay');
      if (overlay) {
        overlay.classList.add('active');
        setTimeout(function() { window.location.href = dest; }, 360);
      } else {
        window.location.href = dest;
      }
    });
  });

  /* ── Smooth transitions on all internal nav links ── */
  document.querySelectorAll('a[href]:not([href^="#"]):not([href^="http"]):not([href^="mailto"]):not([target])').forEach(function(link) {
    if (link.dataset.dest) return;
    link.addEventListener('click', function(e) {
      var overlay = document.getElementById('transition-overlay');
      if (!overlay) return;
      var href = link.getAttribute('href');
      if (!href || href.startsWith('javascript:')) return;
      e.preventDefault();
      overlay.classList.add('active');
      setTimeout(function() { window.location.href = href; }, 320);
    });
  });

  /* ── Animated particles ── */
  var particlesEl = document.getElementById('particles');
  if (particlesEl) {
    for (var i = 0; i < 22; i++) {
      var p    = document.createElement('div');
      p.className = 'particle';
      var size = Math.random() * 4 + 2;
      p.style.cssText =
        'width:' + size + 'px;height:' + size + 'px;' +
        'left:' + (Math.random() * 100) + '%;' +
        'bottom:' + (Math.random() * 40) + '%;' +
        '--pdur:' + (4 + Math.random() * 6) + 's;' +
        '--pdel:' + (Math.random() * 8) + 's;' +
        '--pop:'  + (0.2 + Math.random() * 0.5) + ';' +
        'background:' + (Math.random() > .5 ? 'rgba(0,198,255,0.6)' : 'rgba(96,165,250,0.5)') + ';';
      particlesEl.appendChild(p);
    }
  }

  /* ── Scroll-triggered fade-in ── */
  var scrollObs = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        var delay = entry.target.dataset.scrollDelay || 0;
        setTimeout(function() {
          entry.target.classList.add('scroll-visible');
        }, delay * 100);
        scrollObs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.scroll-reveal').forEach(function(el, i) {
    el.dataset.scrollDelay = el.dataset.scrollDelay || i;
    scrollObs.observe(el);
  });

  /* ── Counter animation ── */
  function animateCount(el) {
    var target = +el.dataset.target;
    var suffix = el.dataset.suffix !== undefined ? el.dataset.suffix : (target >= 99 && target <= 100 ? '%' : '+');
    var dur    = 1800;
    var step   = target / (dur / 16);
    var cur    = 0;
    var t = setInterval(function() {
      cur = Math.min(cur + step, target);
      el.textContent = Math.floor(cur).toLocaleString() + suffix;
      if (cur >= target) clearInterval(t);
    }, 16);
  }

  var counters = document.querySelectorAll('[data-target]');
  if (counters.length) {
    var countObs = new IntersectionObserver(function(entries) {
      if (entries[0].isIntersecting) {
        counters.forEach(animateCount);
        countObs.disconnect();
      }
    }, { threshold: 0.4 });
    countObs.observe(counters[0].parentElement || counters[0]);
  }

  /* ── Smooth anchor scroll ── */
  document.querySelectorAll('a[href^="#"]').forEach(function(a) {
    a.addEventListener('click', function(e) {
      e.preventDefault();
      var target = document.querySelector(a.getAttribute('href'));
      if (target) target.scrollIntoView({ behavior: 'smooth' });
    });
  });

})();
