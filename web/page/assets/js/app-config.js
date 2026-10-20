// 鹏至智库前端服务地址配置（单一数据源）
// 端口约定：
//   KB_PORT_IMPORT = 8100（容器内）-> 导入服务（/upload、/status 任务轮询、/health）
//   KB_PORT_QUERY  = 8101（容器内）-> 问答服务（/query、/stream 流式对话、/history）
//   对外发布时问答服务映射到 8111（见 docker-compose.yml 的 "8111:8101"）
//
// 【重要】主机名必须运行时推导，绝不能硬编码 127.0.0.1：
//   原实现写死 http://127.0.0.1:8101，导致任何「非本机」访问者登录时被
//   location.replace 跳到他自己电脑的 localhost，浏览器报 ERR_CONNECTION_REFUSED
//   （表现为「无法访问此网站」）。改为取 location.hostname 后，公网 IP、域名、
//   内网地址、端口转发均可自适应。
window.KB_SERVICES = (function () {
  var HOST = location.hostname || "127.0.0.1";
  var SCHEME = location.protocol === "https:" ? "https:" : "http:";
  return {
    import: SCHEME + "//" + HOST + ":8100", // 导入服务基地址
    query:  SCHEME + "//" + HOST + ":8111"  // 问答服务基地址（对外 8111）
  };
})();

// 根据元素上的 data-nav 属性，自动把导航链接指向正确的服务端口，
// 避免相对链接在「跨服务」访问时打到错误端口（曾导致「导入失败」）。
//   data-nav="import"   -> 导入服务 /import.html
//   data-nav="chat"     -> 问答服务 /chat.html
//   data-nav="overview" -> 问答服务 /overview.html（概览为静态页，挂靠问答服务即可）
document.addEventListener("DOMContentLoaded", function () {
  var S = window.KB_SERVICES;
  var VER = "20261019b";
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
