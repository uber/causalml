/* Turn an inline contents list into a one-level list whose entries expand.
 *
 * index.rst writes a `maxdepth: 2` toctree into the body of the landing page,
 * which renders every subsection of every page at once.  This adds a toggle in
 * front of each top-level entry that has children and hands the collapsing to
 * collapsible-toctree.css.
 *
 * The toggle is a separate control rather than the entry itself, so following
 * the link still navigates to the page.
 */

(function () {
  "use strict";

  function setup() {
    var wrappers = document.querySelectorAll(".bd-article .toctree-wrapper");

    Array.prototype.forEach.call(wrappers, function (wrapper, wrapperIndex) {
      var items = wrapper.querySelectorAll(":scope > ul > li");
      var collapsible = false;

      Array.prototype.forEach.call(items, function (item, itemIndex) {
        var children = item.querySelector(":scope > ul");
        var link = item.querySelector(":scope > a");

        if (!children || !link) {
          return;
        }

        if (!children.id) {
          children.id = "toctree-children-" + wrapperIndex + "-" + itemIndex;
        }

        var label = link.textContent.trim();
        var toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "toctree-toggle";
        toggle.setAttribute("aria-expanded", "false");
        toggle.setAttribute("aria-controls", children.id);
        toggle.setAttribute("aria-label", "Expand " + label);

        toggle.addEventListener("click", function () {
          var open = item.classList.toggle("is-open");
          toggle.setAttribute("aria-expanded", open ? "true" : "false");
          toggle.setAttribute(
            "aria-label",
            (open ? "Collapse " : "Expand ") + label
          );
        });

        item.insertBefore(toggle, link);
        collapsible = true;
      });

      if (collapsible) {
        wrapper.classList.add("is-collapsible");
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setup);
  } else {
    setup();
  }
})();
