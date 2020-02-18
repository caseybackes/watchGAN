(function(w){var q=window.AmazonUIPageJS||window.P,u=q._namespace||q.attributeErrors,a=u?u("CardJsRuntimeBuzzCopyBuild",""):q;a.guardFatal?a.guardFatal(w)(a,window):a.execute(function(){w(a,window)})})(function(w,q,u){xcp_d("@c/aui-untrusted-ajax",["exports","tslib","@p/a-ajax","@c/guard","@c/logger"],function(a,c,d,b,e){function g(a,k){return b.promise(new Promise(function(b,g){var t=c.__assign(c.__assign({},k),{abort:function(){g("Ajax request aborted")},error:function(c,b,a){g("Ajax request failed with status: "+
b+" and error text: "+a)},success:function(c,a,d){(a=d&&d.http&&d.http.getResponseHeader("Content-Type"))?(a.includes(",")&&(e.log("Ajax response encountered with multiple content-types: "+a+". Defaulting to the first content-type, which could cause problems.","FATAL"),a=a.split(",",1)[0]),a=a.split(";",1)[0]):a="NO-CONTENT-TYPE-FOUND";b({responseBody:c,contentType:a})}});d.ajax(a,t)}))}d=d&&d.hasOwnProperty("default")?d["default"]:d;b=b&&b.hasOwnProperty("default")?b["default"]:b;e=e&&e.hasOwnProperty("default")?
e["default"]:e;var f={contentType:"application/json"};a.default={post:function(a,b,d){b=c.__assign(c.__assign({},f),b);return g(a,{timeout:b.timeout,accepts:b.accepts,contentType:b.contentType,headers:b.additionalHeaders||{},params:d||{},paramsFormat:"json",method:"POST"})}};a.initialize=function(b,a,c){};Object.defineProperty(a,"__esModule",{value:!0})});xcp_d("@c/browser-operations",function(){return{}});xcp_d("@amzn/mix.client-runtime",["exports","tslib"],function(a,c){function d(b){var a=l,c=
new Promise(function(b){a=b}),d=setTimeout(function(){q.P.log("Late loading module "+b,"WARN","MIX")},3E3);c.then(function(){return clearTimeout(d)});return{promise:c,resolve:a}}function b(b){f[b]||(f[b]=d(b));return f[b]}function e(a){return c.__awaiter(this,void 0,void 0,function(){function d(b){b in g||(g[b]=e(b));return g[b]}function e(a){return c.__awaiter(this,void 0,void 0,function(){var e,n,g,k,t;return c.__generator(this,function(c){switch(c.label){case 0:return[4,b(a).promise];case 1:return e=
c.sent(),n=e.capabilities,t=g=e.cardModuleFactory,[4,Promise.all(n.map(d))];case 2:return k=t.apply(void 0,[c.sent()]),f.push(k),[2,k]}})})}var g,f,n;return c.__generator(this,function(b){switch(b.label){case 0:return g={},f=[],[4,Promise.all(a.map(d))];case 1:return n=b.sent(),[2,{requestedOrder:n,initializationOrder:f}]}})})}function g(b){var a="#"===b[0]?b.slice(1):b,a=document.getElementById(a);if(!a)throw Error("Unable to inflate seed ViewModel. No element found for selector: "+b);a=a.dataset.model;
if(!a)return u;try{return JSON.parse(a)}catch(c){throw Error("Unable to inflate seed ViewModel. Unable to parse the model. \nSelector: "+b+". \nValue: "+a);}}var f={},l=function(){};a.registerCapabilityModule=function(a,c){b(a).resolve(c)};a.registerCardFactory=function(b,a){return c.__awaiter(this,void 0,void 0,function(){var d,f,l,n,r,v,x,y;return c.__generator(this,function(c){switch(c.label){case 0:return d=a.capabilities,f=a.cardModuleFactory,[4,e(d)];case 1:return l=c.sent(),n=l.requestedOrder,
r=l.initializationOrder,v=f(n),x=v.P,y=g(b),x.execute(function(){r.forEach(function(a){return a.initialize(b,y,v,x)});v.card(y)}),[2]}})})};Object.defineProperty(a,"__esModule",{value:!0})});xcp_d("@c/dom",["exports"],function(a){var c,d,b={get cardRoot(){return d},get container(){return c}};a.default=b;a.initialize=function(b,a,f){a="#"===b[0]?b.slice(1):b;d=document.getElementById(a);if(!d)throw Error("No node found for selector: "+b);c=d.parentNode;if(d.classList.contains("mix-slot")&&(c=d,d=d.children[0],
!d))throw Error("No card root found within mix-slot");};a.unscope=function(a){return null!==a&&a.__unscope__?a.__unscope__(b):a};Object.defineProperty(a,"__esModule",{value:!0})});xcp_d("@c/guard",["exports","@c/logger"],function(a,c){c=c&&c.hasOwnProperty("default")?c["default"]:c;var d,b=function(b,a){return d.guardFatal(b,a)},e=function(b,a){return d.guardError(b,a)},g=function(b){return d.guardCurrent(b)},f=function(b){return b.catch(function(b){c.log(b.message);throw b;})},l={asFatal:b,asError:e,
current:g,promise:f};a.asError=e;a.asFatal=b;a.current=g;a.default=l;a.initialize=function(b,a,c,e){d=e};a.promise=f;Object.defineProperty(a,"__esModule",{value:!0})});xcp_d("@c/logger",["exports"],function(a){var c,d=function(b,a,d){return c.log(b,a,d)};a.default={log:d};a.initialize=function(b,a,d,f){c=f};a.log=d;Object.defineProperty(a,"__esModule",{value:!0})});xcp_d("@c/remote-operations",["exports","@c/dom","@c/aui-untrusted-ajax","@c/guard","@c/scoped-dom"],function(a,c,d,b,e){function g(b){return function(a){return l(k,
b,a,p)}}function f(b,a){0<a.length&&a.forEach(function(a){b[a]=g(a)})}c=c&&c.hasOwnProperty("default")?c["default"]:c;d=d&&d.hasOwnProperty("default")?d["default"]:d;b=b&&b.hasOwnProperty("default")?b["default"]:b;e=e&&e.hasOwnProperty("default")?e["default"]:e;var l=function(a,c,n,r){a=d.post(a+c,{accepts:"text/html, application/json",contentType:"application/json",additionalHeaders:{"x-amz-acp-params":r}},n);return b.promise(a.then(function(a){var b=a.contentType;a=a.responseBody;if("application/json"===
b)return a||{};if("text/html"===b)try{var c=(new DOMParser).parseFromString(a,"text/html");return e.proxify(c.querySelector("body").firstElementChild)}catch(d){throw Error("Error encountered when parsing html response"+d);}else throw Error("Unexpected content-type found when parsing response: "+b);}))},k,p,m={};a.default={setup:function(a){void 0===a&&(a=[]);f(m,a);return m}};a.initialize=function(a,b,d){if((a=c.cardRoot)&&a.hasAttribute("data-acp-path")&&a.hasAttribute("data-acp-params")){k=a.getAttribute("data-acp-path")||
"";b=a.getAttribute("data-acp-params")||"";try{var e=document.createElement("textarea");e.innerHTML=b;p=0===e.childNodes.length?"":e.childNodes[0].nodeValue||""}catch(g){throw Error("Issue encountered while parsing card attributes when setting up RemoteOperations, error: "+g);}a.removeAttribute("data-acp-path");a.removeAttribute("data-acp-params")}else throw Error("Remote Operation capability requires card root node to exist and have attribute: data-acp-path \x26 data-acp-params");d._operationNames&&
f(m,d._operationNames)};Object.defineProperty(a,"__esModule",{value:!0})});xcp_d("@c/scoped-dom",["exports","tslib","@c/dom"],function(a,c,d){function b(a){return a instanceof HTMLElement||a instanceof Node||a instanceof EventTarget}function e(a){if("undefined"===typeof Proxy||"undefined"===typeof Reflect)return a;var c;c=b(a)?t:a instanceof HTMLCollection||a instanceof NodeList?q:a instanceof Event?m:void 0;var d;d=b(a)?l.cardRoot.contains(a)||!document.contains(a):!0;d?c&&(a.__proxified||(a.__proxified=
new Proxy(a,c)),a=a.__proxified):a=null;return a}function g(a){return a.__proxy||(a.__proxy=function(){for(var b=[],c=0;c<arguments.length;c++)b[c]=arguments[c];b=b.map(function(a){return"function"===typeof a?f(a):d.unscope(a)});return a.apply(d.unscope(this),b)})}function f(a){return a.__proxy||(a.__proxy=function(){for(var b=[],c=0;c<arguments.length;c++)b[c]=arguments[c];return a.apply(e(this),b.map(e))})}var l="default"in d?d["default"]:d,k=Element.prototype.matches||Element.prototype.msMatchesSelector||
Element.prototype.webkitMatchesSelector,p=Element.prototype.closest||function(a){var b=this;do if(k.call(b,a))return b;while(b=b.parentNode);return null},m={get:function(a,b){var c=Reflect.get(a,b);return"__proxified"===b?c:"__unscope__"===b?function(b){return b===l?a:null}:e(c)}},t={get:function(a,b){var d=Reflect.get(a,b);if("__proxified"===b)return d;if("ownerDocument"===b)return u;if("__unscope__"===b)return function(b){return b===l?a:null};"closest"===b&&(d=p);if("function"===typeof d){var k=
d.__proxy;if(!k){if("addEventListener"===b)var m=d,d=function(a,b,d){b="handleEvent"in b?c.__assign(c.__assign({},b),{handleEvent:f(b.handleEvent)}):b;return m.call(this,a,b,d)};k=g(d);d.__proxy=k}return function(){for(var b=[],c=0;c<arguments.length;c++)b[c]=arguments[c];return e(k.apply(a,b))}}return e(d)},set:function(a,b,c){"string"===typeof b&&b.startsWith("on")&&"function"===typeof c?Reflect.set(a,b,function(a){c.call(e(this),e(a))}):Reflect.set(a,b,c);return!0}},q={get:function(a,b){return"number"===
typeof b||"string"===typeof b&&Number.isInteger(Number.parseInt(b,10))?e(Reflect.get(a,b)):Reflect.get(a,b)}};a.default={get cardRoot(){return e(l.cardRoot)},proxify:e};a.initialize=function(a,b,c){};Object.defineProperty(a,"__esModule",{value:!0})});xcp_d("@c/sudo",["exports"],function(a){a.default={get cardRoot(){return null}};a.initialize=function(a,d,b){};a.sudo={};Object.defineProperty(a,"__esModule",{value:!0})});(function(){var a=function(a,b,c){xcp_d(a,b,c)};a.amd=!0;var c,d,b,e,g,f,l,k,p,
m,t,q,n,r,v,x,y,u,w,z;(function(b){function c(a,b){a!==d&&("function"===typeof Object.create?Object.defineProperty(a,"__esModule",{value:!0}):a.__esModule=!0);return function(c,d){return a[c]=b?b(c,d):d}}var d="object"===typeof global?global:"object"===typeof self?self:"object"===typeof this?this:{};"function"===typeof a&&a.amd?a("tslib",["exports"],function(a){b(c(d,c(a)))}):"object"===typeof module&&"object"===typeof module.exports?b(c(d,c(module.exports))):b(c(d))})(function(a){var A=Object.setPrototypeOf||
{__proto__:[]}instanceof Array&&function(a,b){a.__proto__=b}||function(a,b){for(var c in b)b.hasOwnProperty(c)&&(a[c]=b[c])};c=function(a,b){function c(){this.constructor=a}A(a,b);a.prototype=null===b?Object.create(b):(c.prototype=b.prototype,new c)};d=Object.assign||function(a){for(var b,c=1,d=arguments.length;c<d;c++){b=arguments[c];for(var e in b)Object.prototype.hasOwnProperty.call(b,e)&&(a[e]=b[e])}return a};b=function(a,b){var c={},d;for(d in a)Object.prototype.hasOwnProperty.call(a,d)&&0>b.indexOf(d)&&
(c[d]=a[d]);if(null!=a&&"function"===typeof Object.getOwnPropertySymbols){var e=0;for(d=Object.getOwnPropertySymbols(a);e<d.length;e++)0>b.indexOf(d[e])&&Object.prototype.propertyIsEnumerable.call(a,d[e])&&(c[d[e]]=a[d[e]])}return c};e=function(a,b,c,d){var e=arguments.length,g=3>e?b:null===d?d=Object.getOwnPropertyDescriptor(b,c):d,f;if("object"===typeof Reflect&&"function"===typeof Reflect.decorate)g=Reflect.decorate(a,b,c,d);else for(var h=a.length-1;0<=h;h--)if(f=a[h])g=(3>e?f(g):3<e?f(b,c,g):
f(b,c))||g;return 3<e&&g&&Object.defineProperty(b,c,g),g};g=function(a,b){return function(c,d){b(c,d,a)}};f=function(a,b){if("object"===typeof Reflect&&"function"===typeof Reflect.metadata)return Reflect.metadata(a,b)};l=function(a,b,c,d){return new (c||(c=Promise))(function(e,g){function f(a){try{k(d.next(a))}catch(b){g(b)}}function h(a){try{k(d["throw"](a))}catch(b){g(b)}}function k(a){a.done?e(a.value):(new c(function(b){b(a.value)})).then(f,h)}k((d=d.apply(a,b||[])).next())})};k=function(a,b){function c(a){return function(b){return d([a,
b])}}function d(c){if(g)throw new TypeError("Generator is already executing.");for(;e;)try{if(g=1,f&&(h=c[0]&2?f["return"]:c[0]?f["throw"]||((h=f["return"])&&h.call(f),0):f.next)&&!(h=h.call(f,c[1])).done)return h;if(f=0,h)c=[c[0]&2,h.value];switch(c[0]){case 0:case 1:h=c;break;case 4:return e.label++,{value:c[1],done:!1};case 5:e.label++;f=c[1];c=[0];continue;case 7:c=e.ops.pop();e.trys.pop();continue;default:if(!(h=e.trys,h=0<h.length&&h[h.length-1])&&(6===c[0]||2===c[0])){e=0;continue}if(3===c[0]&&
(!h||c[1]>h[0]&&c[1]<h[3]))e.label=c[1];else if(6===c[0]&&e.label<h[1])e.label=h[1],h=c;else if(h&&e.label<h[2])e.label=h[2],e.ops.push(c);else{h[2]&&e.ops.pop();e.trys.pop();continue}}c=b.call(a,e)}catch(k){c=[6,k],f=0}finally{g=h=0}if(c[0]&5)throw c[1];return{value:c[0]?c[1]:void 0,done:!0}}var e={label:0,sent:function(){if(h[0]&1)throw h[1];return h[1]},trys:[],ops:[]},g,f,h,k;return k={next:c(0),"throw":c(1),"return":c(2)},"function"===typeof Symbol&&(k[Symbol.iterator]=function(){return this}),
k};p=function(a,b){for(var c in a)b.hasOwnProperty(c)||(b[c]=a[c])};m=function(a){var b="function"===typeof Symbol&&a[Symbol.iterator],c=0;return b?b.call(a):{next:function(){a&&c>=a.length&&(a=void 0);return{value:a&&a[c++],done:!a}}}};t=function(a,b){var c="function"===typeof Symbol&&a[Symbol.iterator];if(!c)return a;a=c.call(a);var d,e=[],f;try{for(;(void 0===b||0<b--)&&!(d=a.next()).done;)e.push(d.value)}catch(g){f={error:g}}finally{try{d&&!d.done&&(c=a["return"])&&c.call(a)}finally{if(f)throw f.error;
}}return e};q=function(){for(var a=[],b=0;b<arguments.length;b++)a=a.concat(t(arguments[b]));return a};n=function(){for(var a=0,b=0,c=arguments.length;b<c;b++)a+=arguments[b].length;for(var a=Array(a),d=0,b=0;b<c;b++)for(var e=arguments[b],f=0,g=e.length;f<g;f++,d++)a[d]=e[f];return a};r=function(a){return this instanceof r?(this.v=a,this):new r(a)};v=function(a,b,c){function d(a){l[a]&&(p[a]=function(b){return new Promise(function(c,d){1<m.push([a,b,c,d])||e(a,b)})})}function e(a,b){try{var c=l[a](b);
c.value instanceof r?Promise.resolve(c.value.v).then(f,g):k(m[0][2],c)}catch(d){k(m[0][3],d)}}function f(a){e("next",a)}function g(a){e("throw",a)}function k(a,b){(a(b),m.shift(),m.length)&&e(m[0][0],m[0][1])}if(!Symbol.asyncIterator)throw new TypeError("Symbol.asyncIterator is not defined.");var l=c.apply(a,b||[]),p,m=[];return p={},d("next"),d("throw"),d("return"),p[Symbol.asyncIterator]=function(){return this},p};x=function(a){function b(e,f){c[e]=a[e]?function(b){return(d=!d)?{value:r(a[e](b)),
done:"return"===e}:f?f(b):b}:f}var c,d;return c={},b("next"),b("throw",function(a){throw a;}),b("return"),c[Symbol.iterator]=function(){return this},c};y=function(a){function b(d){e[d]=a[d]&&function(b){return new Promise(function(e,f){b=a[d](b);c(e,f,b.done,b.value)})}}function c(a,b,d,e){Promise.resolve(e).then(function(b){a({value:b,done:d})},b)}if(!Symbol.asyncIterator)throw new TypeError("Symbol.asyncIterator is not defined.");var d=a[Symbol.asyncIterator],e;return d?d.call(a):(a="function"===
typeof m?m(a):a[Symbol.iterator](),e={},b("next"),b("throw"),b("return"),e[Symbol.asyncIterator]=function(){return this},e)};u=function(a,b){Object.defineProperty?Object.defineProperty(a,"raw",{value:b}):a.raw=b;return a};w=function(a){if(a&&a.__esModule)return a;var b={};if(null!=a)for(var c in a)Object.hasOwnProperty.call(a,c)&&(b[c]=a[c]);b["default"]=a;return b};z=function(a){return a&&a.__esModule?a:{"default":a}};a("__extends",c);a("__assign",d);a("__rest",b);a("__decorate",e);a("__param",g);
a("__metadata",f);a("__awaiter",l);a("__generator",k);a("__exportStar",p);a("__values",m);a("__read",t);a("__spread",q);a("__spreadArrays",n);a("__await",r);a("__asyncGenerator",v);a("__asyncDelegator",x);a("__asyncValues",y);a("__makeTemplateObject",u);a("__importStar",w);a("__importDefault",z)})})();xcp_d("@c/aui-card",["exports","@p/a-cardui","@p/a-cardui-deck"],function(a,c,d){c=c&&c.hasOwnProperty("default")?c["default"]:c;d=d&&d.hasOwnProperty("default")?d["default"]:d;a.default={getCard:function(a){var d=
c.get(a);return{isExpanded:function(){return d.isExpanded()},toggle:function(){return d.toggle()}}},getCardDeck:function(a){var c=d.get(a);return{initializeAllCards:function(){return c.initializeAllCards()}}}};a.initialize=function(a,c,d){};Object.defineProperty(a,"__esModule",{value:!0})});xcp_d("@c/aui-carousel",["exports","tslib","@c/dom","@p/a-carousel-framework"],function(a,c,d,b){function e(a){var b=this;return function(e,k){return c.__awaiter(b,void 0,void 0,function(){var b;return c.__generator(this,
function(c){switch(c.label){case 0:return[4,a({indexes:e,ids:k})];case 1:b=c.sent();"string"===typeof b&&(b=(new DOMParser).parseFromString(b,"text/html").body.children[0]);if(!b.classList.contains("a-carousel-content-fragment"))throw Error("CarouselRemoteOperation did not return a ContentFragment"+b.innerHTML);return[2,Array.prototype.slice.call(b.querySelectorAll(".a-carousel-card-fragment")).map(d.unscope)]}})})}}b=b&&b.hasOwnProperty("default")?b["default"]:b;a.default={getCarousel:function(a){var c=
b.getCarousel(d.unscope(a)),l=c.getAttr("name");return{gotoPage:function(){return c.gotoPage()},gotoPrevPage:function(){return c.gotoPrevPage()},gotoNextPage:function(){return c.gotoNextPage()},get initialized(){return new Promise(function(a){return b.onInit(l,function(){return a()})})},attachRemoteOperation:function(a){if(c.getAttr("async_provider"))throw Error("Carousel already has attached remoteOperation");c.setAttr("async_provider",e(a))}}}};a.initialize=function(a,b,c){};Object.defineProperty(a,
"__esModule",{value:!0})});xcp_d("@c/aui-truncate",["exports","@c/dom","@p/a-truncate"],function(a,c,d){c=c&&c.hasOwnProperty("default")?c["default"]:c;d=d&&d.hasOwnProperty("default")?d["default"]:d;var b;a.default={updateAll:function(){Array.prototype.slice.call(b).forEach(function(a){return d.get(a).update()})}};a.initialize=function(a,d,f){b=c.cardRoot.getElementsByClassName("a-truncate")};Object.defineProperty(a,"__esModule",{value:!0})});xcp_d("@c/aui-utils",["exports","@p/A","@c/dom"],function(a,
c,d){c=c&&c.hasOwnProperty("default")?c["default"]:c;a.default={hide:function(a){c.hide(d.unscope(a))},show:function(a){c.show(d.unscope(a))},onScreen:function(a,e){return c.onScreen(d.unscope(a),e)},objectIsEmpty:function(a){return c.objectIsEmpty(a)},equals:function(a,d){return c.equals(a,d)},diff:function(a,d){return c.diff(a,d)},throttle:function(a,d,g){return c.throttle(a,d,g)},debounce:function(a,d,g){return c.debounce(a,d,g)},defer:function(a){c.defer(a)},interval:function(a,d){return c.interval(a,
d)},animationFrameDelay:function(a){return c.animationFrameDelay(a)},delay:function(a,d){return c.delay(a,d)},once:function(a){return c.once(a)},attributionChain:function(a){return c.attributionChain(d.unscope(a))}};a.initialize=function(a,c,d){};Object.defineProperty(a,"__esModule",{value:!0})});xcp_d("@c/error-handling",["exports"],function(a){var c,d=function(a,d,g,f){c.error(a,d,g,f)};a.default={error:d};a.error=d;a.initialize=function(a,d,g,f){c=f};Object.defineProperty(a,"__esModule",{value:!0})});
xcp_d("@c/metrics",["exports"],function(a){var c=function(a,c,d){return q.ue.count(a,c,d)},d={count:c};a.count=c;a.default=d;a.initialize=function(a,c,d){};Object.defineProperty(a,"__esModule",{value:!0})});xcp_d("@c/pagemarker",["exports","@p/A","@c/dom","@c/guard"],function(a,c,d,b){function e(a){var d;return function(){return b.promise(d=d||new Promise(function(b){return c.on(a,function(){return b()})}))}}c=c&&c.hasOwnProperty("default")?c["default"]:c;d=d&&d.hasOwnProperty("default")?d["default"]:
d;b=b&&b.hasOwnProperty("default")?b["default"]:b;var g=e("ready"),f=e("load");a.default={get pageReady(){return g()},get pageLoad(){return f()},visible:function(a){void 0===a&&(a=0);var b,e=new Promise(function(a){return b=a}),f=function(){c.onScreen(d.container,a)&&(c.off("scroll resize ready",f),b())};c.on("scroll resize ready",f);f();return e}};a.initialize=function(a,b,c){};Object.defineProperty(a,"__esModule",{value:!0})});xcp_d("@c/x-sample-capability-card-scoped",["exports"],function(a){a.default=
{};a.initialize=function(a,d,b){};Object.defineProperty(a,"__esModule",{value:!0})})});