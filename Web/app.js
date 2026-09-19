/* ============================================================
   Fake Handwriting Detector — front-end logic
   ============================================================ */

const themeToggle = document.getElementById("themeToggle");
const sunIcon = themeToggle.querySelector(".sun-icon");
const moonIcon = themeToggle.querySelector(".moon-icon");
const compareFrame = document.getElementById("compareFrame");

const savedTheme = localStorage.getItem("theme");
const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
const initialTheme = savedTheme || (systemPrefersDark ? "dark" : "light");

themeToggle.addEventListener("click", () => {
  const currentTheme = document.documentElement.getAttribute("data-theme");
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  setTheme(newTheme);
});

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);

  if (theme === "dark") {
    sunIcon.style.display = "none";
    moonIcon.style.display = "block";
  } else {
    sunIcon.style.display = "block";
    moonIcon.style.display = "none";
  }

  // Kirim sinyal perubahan tema ke iframe compare.html
  if (compareFrame && compareFrame.contentWindow) {
    try {
      compareFrame.contentWindow.postMessage({ type: "THEME_CHANGE", theme: theme }, "*");
    } catch (e) {}
  }
}

setTheme(initialTheme);

/* ===== AUTO-RESIZE iframe compare.html =====
   compare.html kirim {type:"FRAME_HEIGHT", height:N} tiap kontennya berubah.
   Parent set tinggi iframe persis setinggi isi → nggak ada space kosong di bawah. */
window.addEventListener("message", (event) => {
  if (event.data && event.data.type === "FRAME_HEIGHT" && typeof event.data.height === "number") {
    if (compareFrame) {
      compareFrame.style.height = Math.ceil(event.data.height) + "px";
    }
  }
});
