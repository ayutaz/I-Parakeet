"""Word lists and carrier sentences for the toy experiment.

* COMMON_WORDS: everyday nouns (native, Sino-Japanese and loanwords) with a
  spread of mora counts and accent types. The toy backbone hears each of them
  many times, so it can read them -> source of self-distillation pairs.
* DIFFICULT_WORDS: rare-kanji words. Any word that shares a character with the
  training text is dropped at build time, so the raw-text backbone has never
  seen how to read them -> evaluation of tags on unseen words.
* Readings and accents come from the OpenJTalk dictionary (pyopenjtalk), which
  is also the oracle that renders references.
"""

COMMON_WORDS = (
    "橋 箸 端 雨 飴 桜 山 川 海 空 花 鳥 犬 猫 魚 肉 水 風 雪 星 月 森 島 道 町 村 駅 店 家 窓 門 机 "
    "椅子 本 紙 箱 鍵 傘 靴 服 帽子 時計 電話 写真 手紙 新聞 雑誌 辞書 地図 切手 野菜 果物 卵 牛乳 "
    "料理 学校 病院 図書館 公園 会社 銀行 先生 学生 医者 友達 家族 子供 兄 姉 弟 妹 朝 夜 春 夏 秋 冬 "
    "天気 音楽 映画 言葉 名前 仕事 旅行 試験 宿題 自転車 飛行機 電車 車 船 部屋 台所 庭 池 石 草 港 "
    "神社 祭り 着物 刀 太鼓 将棋 漢字 歌 声 心 頭 "
    "コーヒー テレビ カメラ ラジオ ピアノ ギター パン バス タクシー ホテル レストラン ニュース ゲーム "
    "サッカー テニス ノート ケーキ メロン バナナ トマト レモン スープ ベッド ドア プール デパート スキー "
    "ボール ハンカチ セーター ネクタイ ポスト"
).split()

DIFFICULT_WORDS = (
    "魑魅魍魎 跋扈 齟齬 蘊蓄 邂逅 揶揄 憂鬱 薔薇 檸檬 葡萄 躊躇 曖昧 顰蹙 杜撰 贔屓 瓦礫 狼狽 咀嚼 痙攣 "
    "鬱蒼 嗚咽 煉瓦 蝙蝠 饂飩 麒麟 駱駝 鸚鵡 蟋蟀 蒟蒻 胡瓜 牡蠣 海鼠 雲雀 蜻蛉 蝸牛 欠伸 鼾 痘痕 罠 鞄 "
    "鋏 錨 蛸 鰻 蕎麦 硝子 茄子 蜜柑 林檎 苺 栗鼠"
).split()

# Carrier sentences; "{}" is a noun heading its own accent phrase.
TEMPLATES = [
    "{}を見ました。",
    "{}が好きです。",
    "昨日、{}について話しました。",
    "あそこに{}があります。",
    "{}の写真を撮りました。",
    "私は{}を探しています。",
    "{}と一緒に帰りました。",
    "新しい{}を買いました。",
    "{}はとても大切です。",
    "今日は{}の話をします。",
    "彼は{}が苦手です。",
    "{}について調べました。",
]
