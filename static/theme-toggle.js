window.onload = function () {
    const savedTheme = localStorage.getItem("theme");

    if (savedTheme) {
        setTheme(savedTheme);
    } else {
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        setTheme(prefersDark ? "dark" : "light");
    }
};

function toggleTheme() {
    const currentTheme = document.getElementById("theme-style").getAttribute("href").includes("darkstyle.css") ? "dark" : "light";
    const newTheme = currentTheme === "light" ? "dark" : "light";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
}

function setTheme(theme) {
    const themeLink = document.getElementById("theme-style");
    const lightHref = themeLink.getAttribute("data-light");
    const darkHref = themeLink.getAttribute("data-dark");

    themeLink.setAttribute("href", theme === "dark" ? darkHref : lightHref);

    const btn = document.getElementById("theme-toggle-btn");
    if (btn) {
        btn.textContent = theme === "dark" ? "🌙 Dark Mode" : "🌞 Light Mode";
    }
}


