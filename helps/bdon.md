# BanG Dream! Our Notes 服务（bdon）

查询 HK/MO/TW 服务器数据，文字使用简体中文；仅支持英文 `/bdon` 前缀指令，命令与参数之间留空白。
管理员使用 `/bdon on`、`/bdon off`、`/bdon status` 独立管理群开关，可与 sekai 同时开启。
角色、乐队、歌曲可使用已配置的别名查询，支持与其他关键词、筛选条件组合使用。

---
### 活动查询
`/bdon event`

不填参数查询当前活动；支持活动 ID、名称关键词或 `上期`。

---
### 活动列表
`/bdon events`

支持活动 ID、名称关键词；使用 `p2` 翻页。

---
### 活动剧情
`/bdon story`

输入活动 ID 或名称关键词。单张自适应长图；顶部标注 AI 生成、请勿转载；`--refresh` 刷新。

---
### 歌曲查询
`/bdon song`

输入歌曲 ID 或名称关键词；支持下方歌曲列表中的难度、等级筛选，例如 `/bdon song ex26`。

---
### 曲绘查询
`/bdon cover`

输入歌曲 ID 或名称关键词，直接发送原图。

---
### 歌曲列表
`/bdon songs`

支持乐队关键词；难度可写 `easy/ez`、`normal/nm`、`hard/hd`、`expert/ex/exp`，大小写不敏感。
等级可写 `26`、`25-28`，也可加 `lv` 前缀；难度与等级可连写为 `ex26`。例如 `/bdon songs ex 26`；使用 `p2` 翻页。

---
### 歌曲效率
`/bdon meta`

支持歌曲 ID、关键词、难度及等级筛选，模式 `free`、`gekisou`；使用 `p2` 翻页。
难度、等级简写同歌曲列表，例如 `/bdon meta ex26`。
排序支持 `sort=efficiency`、`sort=score`、`sort=duration`、`sort=nps`、`sort=coverage`。
同综合力，5×100%/5s 技能，音频时长+30s；默认 `expert`、`gekisou`。

---
### 卡牌查询
`/bdon card`

输入卡牌 ID、名称、角色或乐队关键词；使用 `p2` 翻页。

---
### 卡面查询
`/bdon card art`

输入卡牌 ID 或名称关键词，直接发送完整卡面原图。

---
### 随机漫画
`/bdon comic`

不填参数随机选择，也可输入角色名称筛选，直接发送原图。

---
