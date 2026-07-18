#!/usr/bin/env python3
# -*- coding: utf-8 -*-

ARCHETYPES = {
    "Role-Model": {
        "ocean": "Low N, High E, High C, High A, High O",
        "desc": "Socially desirable, stable, and open-minded.",
    },
    "Self-Centered": {
        "ocean": "Medium N, High E, Low C, Low A, Medium O",
        "desc": "Extroverted but uncooperative and impulsive.",
    },
    "Reserved": {
        "ocean": "Low N, Low E, Medium  C, Medium A, Low O",
        "desc": "Emotionally stable, conscientious, but cautious and conventional.",
    },
    "Average": {
        "ocean": "Medium N, Medium E, Medium C, Medium A, Medium O",
        "desc": "Typical personality without extreme traits, slightly anxious but social.",
    },
}

AGES = ["18~19", "20~39", "40~64", "65+"]

GENDERS = ["Male", "Female"]

# Making vocational choices: A theory of vocational personalities and work environments
JOB_CATEGORIES = {
    "Realistic": "Technical/Manual (e.g., Engineer, Mechanic, Driver)",
    "Investigative": "Analytical/Scientific (e.g., Researcher, Programmer, Analyst)",
    "Artistic": "Creative/Expressive (e.g., Designer, Writer, Musician)",
    "Social": "Helping/Educational (e.g., Teacher, Doctor, Social Worker)",
    "Enterprising": "Leadership/Sales (e.g., Manager, Entrepreneur, Lawyer)",
    "Conventional": "Ordered/Administrative (e.g., Accountant, Librarian, Clerk)",
}

# Culture, leadership, and organizations: The GLOBE study of 62 societies
CULTURAL_CLUSTERS = {
    "Confucian Asia": "China, Japan, South Korea, Singapore",
    "Anglo": "USA, UK, Australia, Canada",
    "Germanic/Nordic Europe": "Germany, Sweden, Netherlands, Norway",
    "Latin America": "Brazil, Mexico, Argentina, Chile",
    "Middle East/Africa": "Egypt, Nigeria, Saudi Arabia, UAE",
    "Southern Asia": "India, Thailand, Vietnam",
}

EN_TO_ZH_MAP = {
    "Role-Model": "榜样型",
    "Low N, High E, High C, High A, High O": "低N，高E，高C，高A，高O",
    "Socially desirable, stable, and open-minded.": "理想的，稳定的，思想开放的。",
    "Self-Centered": "自我为中心型",
    "Medium N, High E, Low C, Low A, Medium O": "中N，高E，低C，低A，中O",
    "Extroverted but uncooperative and impulsive.": "性格外向，但不合作，冲动。",
    "Reserved": "保守型",
    "Low N, Low E, Medium  C, Medium A, Low O": "低N，低E，中C，中A，低O",
    "Emotionally stable, conscientious, but cautious and conventional.": "情绪稳定，有责任心，但谨慎保守。",
    "Average": "平均型",
    "Medium N, Medium E, Medium C, Medium A, Medium O": "中等N，中等E，中等C，中等A，中等O",
    "Typical personality without extreme traits, slightly anxious but social.": "典型的性格，没有极端的特征，有点焦虑，但善于交际。",
    "Male": "男性",
    "Female": "女性",
    "18~19": "18~19",
    "20~39": "20~39",
    "40~64": "40~64",
    "65+": "65+",
    "Realistic": "现实的",
    "Technical/Manual (e.g., Engineer, Mechanic, Driver)": "技术/手册（如工程师、机械师、司机）",
    "Investigative": "调查",
    "Analytical/Scientific (e.g., Researcher, Programmer, Analyst)": "分析/科学（例如，研究员、程序员、分析师）",
    "Artistic": "艺术性",
    "Creative/Expressive (e.g., Designer, Writer, Musician)": "创意/表现力（如设计师、作家、音乐家）",
    "Social": "社会性",
    "Helping/Educational (e.g., Teacher, Doctor, Social Worker)": "帮助/教育（如教师、医生、社工）",
    "Enterprising": "有事业心的",
    "Leadership/Sales (e.g., Manager, Entrepreneur, Lawyer)": "领导/销售（如经理、企业家、律师）",
    "Conventional": "传统的",
    "Ordered/Administrative (e.g., Accountant, Librarian, Clerk)": "命令/行政（如会计、图书管理员、文员）",
    "Confucian Asia": "儒家的亚洲",
    "China, Japan, South Korea, Singapore": "中国，日本，韩国，新加坡",
    "Anglo": "英美资源集团",
    "USA, UK, Australia, Canada": "美国，英国，澳大利亚，加拿大",
    "Germanic/Nordic Europe": "北欧日耳曼/欧洲",
    "Germany, Sweden, Netherlands, Norway": "德国，瑞典，荷兰，挪威",
    "Latin America": "拉丁美洲",
    "Brazil, Mexico, Argentina, Chile": "巴西，墨西哥，阿根廷，智利",
    "Middle East/Africa": "中东/非洲",
    "Egypt, Nigeria, Saudi Arabia, UAE": "埃及，尼日利亚，沙特阿拉伯，阿联酋",
    "Southern Asia": "南亚",
    "India, Thailand, Vietnam": "印度，泰国，越南",
}

ATOMIC_DIMENSIONS = [
    "xAttr",
    "xReact",
    "xWant",
    "xNeed",
    "xIntent",
    "xEffect",
    "oWant",
    "oReact",
    "oEffect",
]

# =============================================================================
# Step 6: Alpaca Dataset Building
# =============================================================================

INSTRUCTION_PROMPT = "[Task Type]\n{TaskType}\n\n[Task Input]\n{TaskInput}"
SYSTEM_PROMPT = "You are a personalized AI assistant. Persona:\n{Persona}"
RELATION_TO_QUESTION = {
    "xAttr": "How would you describe yourself?",
    "xWant": "What will you do after the event?",
    "xNeed": "What do you need to do before the event?",
    "xIntent": "What was the motive that triggered the incident?",
    "xEffect": "What impact will the occurrence of the incident bring?",
    "xReact": "How do you feel when the incident occurs?",
    "oEffect": "What impact will the event have on others?",
    "oWant": "What will others do if the incident occurs?",
    "oReact": "How did others feel when the incident occurred?",
}
