// 鹏至智库前端服务地址配置（单一数据源）
// 必须与后端 .env 的端口保持一致：
//   KB_PORT_IMPORT = 8100  -> 导入服务（/upload、/status 任务轮询、/health）
//   KB_PORT_QUERY  = 8101  -> 问答服务（/query、/stream 流式对话、/history）
// 说明：8000/8001 已被「清衡舆情分析平台」占用，本项目统一后移 100，故用 8100/8101。
// 若后续端口再次调整，只需修改本文件的两个地址即可，前端所有调用与导航会自动跟随。
window.KB_SERVICES = {
  import: "http://127.0.0.1:8100", // 导入服务基地址
  query:  "http://127.0.0.1:8101"  // 问答服务基地址
};

// 根据元素上的 data-nav 属性，自动把导航链接指向正确的服务端口，
// 避免相对链接在「跨服务」访问时打到错误端口（曾导致「导入失败」）。
//   data-nav="import"   -> 导入服务 /import.html
//   data-nav="chat"     -> 问答服务 /chat.html
//   data-nav="overview" -> 问答服务 /overview.html（概览为静态页，挂靠问答服务即可）
document.addEventListener("DOMContentLoaded", function () {
  var S = window.KB_SERVICES;
  var VER = "20261013";
  // 页面版本号：用于强制浏览器丢弃旧缓存（与各页脚 v1.0.0+<date> 对应）。
  // 所有内部跳转都带 ?v=，保证每次导航落到一个「从未加载过」的 URL，拿到的必定是最新页面。

  // 会话态读取：localStorage / sessionStorage 任一存在即视为已登录（前端演示，无后端校验）。
  function getLogin() {
    try { return localStorage.getItem('kb_logged_in') || sessionStorage.getItem('kb_logged_in'); }
    catch (e) { return null; }
  }
  // 受保护页面守卫：登录页本身不拦截；其余页面未登录则跳转登录页，并携带当前地址以便登录后返回。
  var path = location.pathname;
  if (path.indexOf('login.html') === -1) {
    if (!getLogin()) {
      var here = location.pathname + location.search;
      location.replace(S.import + "/login.html?v=" + VER + "&next=" + encodeURIComponent(here));
      return;
    }
    // 已登录：把应用栏与用户菜单中的显示名回填为登录账号
    try {
      var info = JSON.parse(getLogin());
      if (info && info.user) {
        var uName = document.querySelector('.appbar__uname');
        if (uName) uName.textContent = info.user;
        var hd = document.querySelector('.appbar__menu__hd b');
        if (hd) hd.textContent = info.user;
        var av = document.querySelector('.appbar__avatar');
        if (av && info.user) av.textContent = info.user.slice(0, 1);
      }
    } catch (e) {}
  }

  // data-nav 映射表：import/chat/overview 指向对应服务页面；portal 指向导入服务根路径（门户着陆页）。
  var nodes = document.querySelectorAll("[data-nav]");
  nodes.forEach(function (a) {
    var key = a.getAttribute("data-nav");
    if (key === "portal") {
      a.setAttribute("href", S.import + "/?v=" + VER);
    } else if (key === "import" || key === "chat" || key === "overview") {
      var base = key === "import" ? S.import : S.query;
      a.setAttribute("href", base + "/" + key + ".html?v=" + VER);
    }
  });

  // 全局搜索框：回车后跳转到智能问答页并带上查询词（?q=），由 chat.html 自动预填。
  var search = document.querySelector(".appbar__search input");
  if (search) {
    search.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        var q = encodeURIComponent((search.value || "").trim());
        location.href = S.query + "/chat.html?v=" + VER + (q ? ("&q=" + q) : "");
      }
    });
  }

  // 全局用户菜单：点击头像展开/收起，点击外部自动关闭；退出登录仅清除本地会话并回到门户。
  var userBtn = document.getElementById("userBtn");
  var userMenu = document.getElementById("userMenu");
  if (userBtn && userMenu) {
    userBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      userMenu.hidden = !userMenu.hidden;
    });
    document.addEventListener("click", function (e) {
      if (!userMenu.hidden && e.target !== userBtn && !userMenu.contains(e.target)) {
        userMenu.hidden = true;
      }
    });
    var btnLogout = document.getElementById("btnLogout");
    if (btnLogout) {
      btnLogout.addEventListener("click", function () {
        try {
          localStorage.removeItem("kb_logged_in");
          sessionStorage.removeItem("kb_logged_in");
          localStorage.removeItem("kb_session_id");
        } catch (e) {}
        userMenu.hidden = true;
        location.href = S.import + "/login.html?v=" + VER;
      });
    }
  }
});
