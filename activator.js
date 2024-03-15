'use strict';

var raycastBackendURL = typeof $argument != "undefined" ? $argument : 'https://custome-backend.self.com'

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
        url: $request.url.replace('https://backend.raycast.com', raycastBackendURL),
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
                    continue;
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