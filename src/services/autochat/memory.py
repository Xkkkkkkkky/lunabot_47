from .utils import *
import time
import os
from typing import List, Tuple
from dataclasses import dataclass, field, asdict

@dataclass
class UserMemory:
    names: List[str] = field(default_factory=list)
    profile: str = ""
    recent_events: List[Tuple[float, str]] = field(default_factory=list)

@dataclass
class SelfMemory:
    id: str
    text: str
    time: float


class MemorySystem:
    def __init__(self, data_dir: str, group_id: int):
        os.makedirs(data_dir, exist_ok=True)
        self.file_db = get_file_db(os.path.join(data_dir, f"memory_{group_id}.json")) 

    def um_update(
        self, 
        user_id: int, 
        new_names: List[str] = None,
        wrong_names: List[str] = None,
        profile_update: str = None, 
        event_update: str = None, 
        max_events: int = 5, 
        max_names: int = 5,
    ):
        """
        更新用户记忆（增量更新）。
        """
        ums = self.file_db.get('ums', {})
        uid_str = str(user_id)
        
        # 加载现有记忆或创建新的
        if uid_str in ums:
            current_um = UserMemory(**ums[uid_str])
            if not isinstance(current_um.names, list): current_um.names = list(current_um.names)
            if not isinstance(current_um.recent_events, list): current_um.recent_events = list(current_um.recent_events)
        else:
            current_um = UserMemory()

        updated = False
        
        # 1. 更新名字
        if wrong_names:
            for wrong_name in wrong_names:
                if wrong_name in current_um.names:
                    current_um.names.remove(wrong_name)
                    updated = True
                    info(f"移除用户 {user_id} 错误名字: {wrong_name}")
        if new_names:
            for new_name in new_names:
                if new_name in current_um.names:
                    continue
                current_um.names.append(new_name)
                current_um.names = current_um.names[-max_names:]
                updated = True
                info(f"更新用户 {user_id} 曾用名: {new_name}")
        
        # 2. 更新用户画像
        if profile_update and profile_update != current_um.profile:
            current_um.profile = profile_update
            updated = True
            info(f"更新用户 {user_id} 画像")

        # 3. 添加新事件 
        if event_update:
            current_um.recent_events.append((time.time(), event_update))
            current_um.recent_events = current_um.recent_events[-max_events:]
            updated = True
            info(f"更新用户 {user_id} 事件: {event_update}")

        if updated:
            ums[uid_str] = asdict(current_um)
            self.file_db.set('ums', ums)

    def um_get(self, user_id: int) -> UserMemory | None:
        ums = self.file_db.get('ums', {})
        if str(user_id) in ums:
            data = ums[str(user_id)]
            # 兼容旧数据结构（如果之前只存了text）
            if 'text' in data and 'profile' not in data:
                 return UserMemory(profile=data['text'])
            return UserMemory(**data)
        return None
    
    def um_query_uid_by_name_in_message(self, message: str) -> set[int]:
        """
        通过消息内容中出现的名字查询用户记忆对应的用户ID。
        """
        ums = self.file_db.get('ums', {})
        results = set()
        for uid_str, data in ums.items():
            user_memory = UserMemory(**data)
            for name in user_memory.names:
                if name in message:
                    results.add(int(uid_str))
                    break
        return results

    def sm_add(self, msg_id: int, text: str, keep_count: int):
        """
        更新自身对话记忆。

        参数:
            msgs (list[dict]): 新加入的对话消息列表。
            keep_count (int): 保留的消息数量。
        """
        sms = self.file_db.get('sms', [])
        sms.append({
            'id': str(msg_id),
            'text': text,
            'time': datetime.now().timestamp(),
        })
        sms = sms[-keep_count:]
        self.file_db.set('sms', sms)
        info(f"更新自身对话记忆，保留最近 {keep_count} 条消息")

    def sm_get(self) -> list[SelfMemory]:
        """
        获取自身对话记忆。

        返回:
            list[dict]: 对话消息列表。
        """
        return [SelfMemory(**sm) for sm in self.file_db.get('sms', [])]


