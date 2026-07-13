/* ============================================================
   Fake Handwriting Detector — front-end logic
   The app itself is now just a hub linking to two tools:
     - "Gambar Tanda Tangan" -> Voila (canvas drawing + compare)
     - "Upload & Bandingkan Foto" -> compare.html, embedded (Opsi A)
   The only remaining logic here is the light/dark theme toggle.
   ============================================================ */

const themeToggle = document.getElementById("themeToggle");
const sunIcon = themeToggle.querySelector(".sun-icon");
const moonIcon = themeToggle.querySelector(".moon-icon");

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
}

setTheme(initialTheme);
