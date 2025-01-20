import requests
import importlib.metadata
from typing import Union

from error.utilserr import PackageNotFoundFronPypiError

def get_installed_version(package_name: str = "catena") -> Union[str, None]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError(package_name) as e:
        print(e)
        return None

def get_versions_from_pypi(package_name: str = "catena"):
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    try:
        if response.status_code == 200:
            data = response.json()
            versions = list(data['releases'].keys())
            return versions
    except PackageNotFoundFronPypiError(code = response.status_code) as e:
        print(e)

def check_update(package_name: str = "catena") -> bool:

    installed_version = get_installed_version(package_name)
    versions_from_pypi = get_versions_from_pypi(package_name)
    
    if installed_version is not None and versions_from_pypi is not None:
        latest_version = versions_from_pypi[-1]
        if installed_version != latest_version:
            print(f"[check_update] A new version of {package_name} is available: {latest_version}")
            return True
    else:
        print(f"[check_update] Check update failed.")
        return False
        
def get_package_info(package_name: str = "catena"):
    # 向 PyPI 的 API 发送请求
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    if response.status_code == 200:
        # 解析 JSON 数据
        package_info = response.json()
        # 获取 'info' 部分
        info = package_info.get('info')
        if info:
            # 获取 'attrs' 部分
            attrs = info.get('attrs')
            if attrs:
                # 提取 'note' 属性
                note = attrs.get('note')
                return note
    return None