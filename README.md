# Unlocking-Raycast-With-Surge

**写在前面：在使用之前，请明确这是破解行为，是不值得被提倡的。我之所以仍然保留此仓库，原因是我认为现有的代码的所有 AI 功能均为自己的 Key 实现，尚且没有同 Raycast 出现明确的利益冲突。**

利用 Surge 的 MiTM 功能拦截请求，并利用 Docker 服务模拟后端操作，从而实现 Raycast 的激活。

Docker 服务是一个简单的 [Raycast](https://raycast.com/) 的 API 代理。它允许您在不订阅的情况下使用包括 [Raycast AI](https://raycast.com/ai) 、翻译、同步在内的 Pro 功能（但是都需要你拥有自己的 API Key）。实现原理如下：

-   Raycast Pro：在服务端修改 `/me`、`/ai/models` 等关键请求的返回字段。
-   Raycast AI：对于非 AI Completions 之外的某些请求的返回值修改关键字段，而对于 AI Completions 的请求，将 Raycast 的请求转换格式转发到 OpenAI 的 API，响应后再实时转换格式返回。
-   Translator：目前的翻译功能实现也是基于 OpenAI 的，会很慢，你可以自行换成别的商业 API 。
-   Sync：基于本地 JSON 处理，没有用到数据库，避免了额外的配置，但是会有一定的性能损失。

Docker 服务修改自 [yufeikang/raycast_api_proxy](https://github.com/yufeikang/raycast_api_proxy)。由于我在迁移的时候出现了问题，新建此仓库后失去了 `fork` 的属性。但保留了 `.git` 贡献历史。

在开始之前，请确保你拥有以下条件：

-   Surge 激活版本
-   一台云服务器，拥有 Docker、公网 IP
-   一个域名
-   一个 Open AI 格式的 API Key 与对应的 Base URL

本文用到的的服务器配置为：Ubuntu 22.04，1Panel + OpenResty + Docker

鉴于会折腾 MiTM 的朋友应该是有一定基础的，本文我主要给出具体的配置文件，还有自己的一些踩坑。

整体的配置思路如下：

Raycast - Surge MiTM 覆写请求 URL - 云服务器 - Nginx 反向代理 - Docker 后端。

## 使用方法

### 在服务端启动后端服务

由于 Raycast 的 AI 功能请求与返回都进行了包装，直接请求类似 OpenAI 的接口并不可行，需要使用一个转接器来模拟 Raycast 的后端转接格式，这就需要我们自行搭建并运行后端。

#### 基于 Docker 的方法

此 Docker 镜像修改自 [yufeikang/raycast_api_proxy](https://github.com/yufeikang/raycast_api_proxy)，修复了一些迁移到服务器上会导致的问题。

```bash
docker run --name Raycast \
    -e OPENAI_API_KEY=sk-xxxx \
    -e OPENAI_BASE_URL=https://api.openai.com/v1/ \
    -p 12443:80 -e LOG_LEVEL=INFO -d arthals/raycast-backend:latest
```

这会在你的服务器上的 `12443` 端口启动一个 Raycast 的 API 代理服务，你需要自行修改 `OPENAI_API_KEY` 与 `OPENAI_BASE_URL`。

这里存在两处需要注意的地方：

1. 原仓库给的示例是 `OPENAI_API_BASE`，但是由于 `openai` 库升级了，所以现在需要使用 `OPENAI_BASE_URL`。
2. 原仓库给的示例有一个 `--dns 1.1.1.1` 的配置，但是由于众所周知的原因，腾讯云的服务器无法访问这个 DNS 服务器，所以你需要移除这个配置。

注意 OPENAI_BASE_URL 的计算方式如下：

如果你的 API 请求的 URL 是 https://api-proxy.com/v1/chat/completions ，那么就是 https://api-proxy.com/v1/

为了使用此代理，你还需要参照 https://arthals.ink/posts/coding/unlocking-raycast-with-surge 或按照下文设置 Surge MiTM 与服务器 Nginx 配置以支持流式传输。

SSL 证书直接通过 1Panel 的 Nginx（Openresty）搞定就行。与 Surge 和服务器后端均无关。

本仓库只支持服务器使用，本地使用需要能自行签发 SSL 证书。若本地使用建议参照原仓库操作。

目前还有如下环境变量可以使用：

* `DEEPLX_API_KEY`：DeepLX 的 API Key，用于翻译功能。
* `DEEPLX_BASE_URL`：DeepLX 的 API URL，用于翻译功能。
* `APYHUB_API_KEY`：[ApyHub](https://apyhub.com/utility/serp-rank) 的 API Key，用于联网搜索的 Function Call。

#### 基于 Python + PM2 的方法

你亦可以简单的使用 Python 和 PM2 来启动服务：

```bash
# 安装依赖
pip install -r requirements.txt
# 启动服务
pm2 start 'OPENAI_API_KEY="sk-xxxx" OPENAI_BASE_URL="https://api.openai.com/v1/" uvicorn app.main:app --host 0.0.0.0 --port 12443 --reload' --name Raycast
```

### Nginx 服务端配置

按照正常的 1Panel + OpenResty 配置一个反向代理网站，将 `https://custome-backend.self.com/` 代理到 `http://127.0.0.1:12443/` 即可。

特别地，你需要修改反向代理的配置文件，移除默认的压缩算法以支持流式传输，你可以参见：https://github.com/lobehub/lobe-chat/discussions/531

修改后的反向代理配置文件示例如下：

```nginx
location ^~ / {
    proxy_pass http://127.0.0.1:12443;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header REMOTE-HOST $remote_addr;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_http_version 1.1;
    add_header Cache-Control no-cache;
    proxy_cache off;  # 关闭缓存
    proxy_buffering off;  # 关闭代理缓冲
    chunked_transfer_encoding on;  # 开启分块传输编码
    tcp_nopush on;  # 开启TCP NOPUSH选项，禁止Nagle算法
    tcp_nodelay on;  # 开启TCP NODELAY选项，禁止延迟ACK算法
    keepalive_timeout 300;  # 设定keep-alive超时时间为65秒
}
```

### Surge MiTM 配置与激活脚本

本质是劫持 Raycast 的请求，将请求转发到服务器上。配置如下：

```conf
# surge.conf
[MITM]
skip-server-cert-verify = true
h2 = true
hostname = *.raycast.com
ca-passphrase = ...
ca-p12 = MIIKP...

[Script]
raycast-activate-backend.raycast.com = type=http-request,pattern=^https://backend.raycast.com,max-size=0,debug=1,script-path=activator.js
```

请注意，你需要将 `ca-passphrase` 与 `ca-p12` 替换为你自己的 Surge CA 密码与证书。

其中用到的 `activator.js` 脚本修改自 [wibus-wee/activation-script](https://github.com/wibus-wee/activation-script/)，你需要自行修改其中的 `https://custome-backend.self.com` 为你的服务器地址。

请将 `activator.js` 放置在 Surge 配置文件夹下。

Activator.js 内容如下（亦可在本仓库找到）：

```js
'use strict';

function transformToString(obj) {
    if (typeof obj === 'object') {
        return JSON.stringify(obj);
    }
    return obj;
}
/**
 * 构建 Surge 响应体
 *
 * @param props 响应体属性
 * @description 该函数将会自动将对象转换为 JSON 字符串，因此你可以直接传入对象
 */
function buildResponse(props) {
    if (props.body) {
        props.body = transformToString(props.body);
    }
    // console.log(props.body);
    $done({
        ...props,
    });
}
/**
 * 发送通知
 *
 * @param title 标题
 * @param subtitle 副标题
 * @param body 内容
 * @description 该函数将会自动将对象转换为 JSON 字符串，因此你可以直接传入对象
 */
function sendNotification(title, subtitle, body) {
    title = transformToString(title);
    subtitle = transformToString(subtitle);
    body = transformToString(body);
    $notification.post(title, subtitle, body);
}
const methods = ['get', 'put', 'delete', 'head', 'options', 'patch', 'post'];
/**
 * 发送请求
 * @param props 请求参数
 * @param callback 回调函数
 */
const httpClient = {};
for (let method of methods) {
    // @ts-ignore
    httpClient[method] = (props, callback) => {
        $httpClient[method](props, callback);
    };
}

/**
 * @url https://backend.raycast.com/api/
 */
function raycastActivate() {
    $done({
        url: $request.url.replace('https://backend.raycast.com', 'https://custome-backend.self.com'),
        headers: $request.headers,
        body: $request.body,
    });
}

const activator = {
    raycast: {
        base: 'https://backend.raycast.com/api/',
        activate: [
            {
                base: '*',
                func: raycastActivate,
            },
        ],
    },
};

const url = $request.url;
/**
 * Determine whether the URL matches the base
 */
function isMatchBase(url, base) {
    if (Array.isArray(base)) {
        for (let item of base) {
            if (url.includes(item)) {
                return true;
            }
        }
        return false;
    }
    return url.includes(base);
}
/**
 * Automatic execution of the corresponding function according to the URL
 */
function launch() {
    for (let module in activator) {
        if (isMatchBase(url, activator[module].base)) {
            for (let key in activator[module]) {
                if (key === 'base') continue;
                if (Array.isArray(activator[module][key])) {
                    for (let custom of activator[module][key]) {
                        // 检查 custom.base 是否为通配符 '*'，如果是，则匹配任何以 activator[module].base 开头的URL
                        if (custom.base === '*' && url.startsWith(activator[module].base)) {
                            return custom.func();
                        }
                        // 否则，检查精确匹配
                        else if (url === `${activator[module].base}/${custom.base}`) {
                            return custom.func();
                        }
                    }
                    continue;
                }
                if (typeof activator[module][key] === 'object') {
                    // 检查是否为通配符 '*'，如果是，则匹配任何以 activator[module].base 开头的URL
                    if (activator[module][key].base === '*' && url.startsWith(activator[module].base)) {
                        return activator[module][key].func();
                    }
                    if (url === `${activator[module].base}/${activator[module][key].base}`) {
                        return activator[module][key].func();
                    }
                } else if (!url.includes(`${activator[module].base}/${key}`)) {
                    return;
                }
                if (typeof activator[module][key] === 'function') {
                    return activator[module][key]();
                }
            }
        }
    }
    console.log(`[activator] ${url} is not matched`);
    returnDefaultResponse();
    $done();
    return;
}
function returnDefaultResponse() {
    console.log(`[activator] returnDefaultResponse: ${url} - ${$request.method.toLowerCase()}`);
    // @ts-expect-error
    httpClient[$request.method.toLowerCase()](
        {
            url: $request.url,
            headers: $request.headers,
        },
        (err, response, _data) => {
            if (!_data) {
                console.log('returnDefaultResponse: _data is null', err);
                buildResponse({
                    status: 500,
                    body: {},
                });
            }
            buildResponse({
                status: response.status,
                headers: response.headers,
                body: _data,
            });
        }
    );
}

launch();
```

### 在本地启动后端服务

如果你想要在本地使用这个服务，那么你需要设置 httpx 的 verify 为 false，然后为了避免后端请求 backend.raycast.com 又请求回自身，你还需要修改 pass_through 函数，额外设置一个私有的 header，然后再在 Surge script 里判断一下，这样做就可以不做自签名证书使用了。

例如：

```python
# app/main.py
# 修改 http_client = httpx.AsyncClient() 为
http_client = httpx.AsyncClient(verify=False)

# app/utils.py
# 修改 pass_through 函数，原先为：
"""
async def pass_through_request(client: httpx.AsyncClient, request: ProxyRequest):
    ...
    headers["accept-encoding"] = "identity"
    try:
        response = await client.request(
            request.method,
            url,
            headers=headers,
            data=request.body,
            params=request.query_params,
            timeout=60.0,
        )
    ...
"""
# 修改后为：
async def pass_through_request(client: httpx.AsyncClient, request: ProxyRequest):
    ...
    headers["accept-encoding"] = "identity"
    headers["arthals"] = "arthals"
    try:
        response = await client.request(
            request.method,
            url,
            headers=headers,
            data=request.body,
            params=request.query_params,
            timeout=60.0,
        )
    ...
```

对于 activator.js，你需要修改为：

```js
function raycastActivate() {
    // 检查 $request.headers 是否有 arthals 字段
    if ($request.headers['arthals']) {
        $done({});
        return;
    }
    $done({
        url: $request.url.replace('https://backend.raycast.com', 'http://localhost:12443'),
        headers: $request.headers,
        body: $request.body,
    });
}
```

## Credit

[wibus-wee/activation-script](https://github.com/wibus-wee/activation-script)

[yufeikang/raycast_api_proxy](https://github.com/yufeikang/raycast_api_proxy)
