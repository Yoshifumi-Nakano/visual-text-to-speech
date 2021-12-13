import os

#音素一文字で一つのひらがなのリスト
oneWordByVowel=["a","i","u","e","o","q","N","sp"]

#音素二つで二つのひらがなのリスト
twoWordByTwoConsonant=[
    "kya",
    "kyi",
    "kyu",
    "kye",
    "kyo",
    "sya",
    "syu",
    "syo",
    "sha",
    "shu",
    "she",
    "sho",
    "tya",
    "tyu",
    "tyo",
    "cha",
    "chu",
    "che",
    "cho",
    "nya",
    "nyi",
    "nyu",
    "nye",
    "nyo",
    "fa",
    "fi",
    "fe",
    "fo",
    "hya",
    "hyu",
    "hyo",
    "mya",
    "myi",
    "myu",
    "mye",
    "myo",
    "ye",
    "rya",
    "ryi",
    "ryu",
    "rye",
    "ryo",
    "wi",
    "we",
    "gya",
    "gyi",
    "gyu",
    "gye",
    "gyo",
    "zya",
    "zyu",
    "zyo",
    "ja",
    "ju",
    "je",
    "jo",
    "bya",
    "byu",
    "byo",
    "pya",
    "pyu",
    "pyo",
    "we",
    "tsa",
    "tsi",
    "tse",
    "tso",
    "va",
    "vi",
    "ve",
    "vo",
    "dya",
    "dyu",
    "dye",
    "dyo"
    ]

#音素とひらがなの対応表
dic={
    "a":"あ",
    "i":"い",
    "u":"う",
    "e":"え",
    "o":"お",
    "N":"ん",
    "q":"っ",
    "sp":"、",
    "ka":"か",
    "ki":"き",
    "ku":"く",
    "ke":"け",
    "ko":"こ",
    "kya":"きゃ",
    "kyi":"きぃ",
    "kyu":"きゅ",
    "kye":"きぇ",
    "kyo":"きょ",
    "sa":"さ",
    "si":"し",
    "shi":"し",
    "su":"す",
    "se":"せ",
    "so":"そ",
    "sya":"しゃ",
    "syu":"しゅ",
    "syo":"しょ",
    "sha":"しゃ",
    "shu":"しゅ",
    "she":"しぇ",
    "sho":"しょ",
    "ta":"た",
    "ti":"ち",
    "tu":"つ",
    "tsu":"つ",
    "te":"て",
    "to":"と",
    "tya":"ちゃ",
    "tyu":"ちゅ",
    "tyo":"ちょ",
    "cha":"ちゃ",
    "chi":"ち",
    "chu":"ちゅ",
    "che":"ちぇ",
    "cho":"ちょ",
    "na":"な",
    "ni":"に",
    "nu":"ぬ",
    "ne":"ね",
    "no":"の",
    "nya":"にゃ",
    "nyi":"にぃ",
    "nyu":"にゅ",
    "nye":"にぇ",
    "nyo":"にょ",
    "ha":"は",
    "hi":"ひ",
    "hu":"ふ",
    "fu":"ふ",
    "he":"へ",
    "ho":"ほ",
    "fa":"ふぁ",
    "fi":"ふぃ",
    "fe":"ふぇ",
    "fo":"ふぉ",
    "hya":"ひゃ",
    "hyu":"ひゅ",
    "hyo":"ひょ",
    "ma":"ま",
    "mi":"み",
    "mu":"む",
    "me":"め",
    "mo":"も",
    "mya":"みゃ",
    "myi":"みぃ",
    "myu":"みゅ",
    "mye":"みぇ",
    "myo":"みょ",
    "ya":"や",
    "yu":"ゆ",
    "ye":"いぇ",
    "yo":"よ",
    "ra":"ら",
    "ri":"り",
    "ru":"る",
    "re":"れ",
    "ro":"ろ",
    "rya":"りゃ",
    "ryi":"りぃ",
    "ryu":"りゅ",
    "rye":"りぇ",
    "ryo":"りょ",
    "wa":"わ",
    "wi":"うぃ",
    "wu":"う",
    "we":"うぇ",
    "wo":"を",
    "ga":"が",
    "gi":"ぎ",
    "gu":"ぐ",
    "ge":"げ",
    "go":"ご",
    "gya":"ぎゃ",
    "gyi":"ぎぃ",
    "gyu":"ぎゅ",
    "gye":"ぎぇ",
    "gyo":"ぎょ",
    "za":"ざ",
    "zi":"じ",
    "zu":"ず",
    "ze":"ぜ",
    "zo":"ぞ",
    "zya":"じゃ",
    "zyu":"じゅ",
    "zyo":"じょ",
    "ja":"じゃ",
    "ji":"じ",
    "ju":"じゅ",
    "je":"じぇ",
    "jo":"じょ",
    "da":"だ",
    "di":"ぢ",
    "du":"づ",
    "de":"で",
    "do":"ど",
    "ba":"ば",
    "bi":"び",
    "bu":"ぶ",
    "be":"べ",
    "bo":"ぼ",
    "bya":"びゃ",
    "byu":"びゅ",
    "byo":"びょ",
    "pa":"ぱ",
    "pi":"ぴ",
    "pu":"ぷ",
    "pe":"ぺ",
    "po":"ぽ",
    "pya":"ぴゃ",
    "pyu":"ぴゅ",
    "pyo":"ぴょ",
    "we":"うぇ",
    "tsa":"つぁ",
    "tsi":"つぃ",
    "tsu":"つ",
    "tse":"つぇ",
    "tso":"つぉ",
    "va":"ゔぁ",
    "vi":"ゔぃ",
    "vu":"ゔ",
    "ve":"ゔぇ",
    "vo":"ゔぉ",
    "dya":"ぢゃ",
    "dyu":"ぢゅ",
    "dye":"ぢぇ",
    "dyo":"ぢょ"
}

#きゃ、きゅ、きょなどは2文字のひらがなとして扱う
def Phoneme2Kana_ver2(phoneme,duration):
    kana=[]
    kanaDuration=[]
    N=len(phoneme)
    i=0
    while i<N:
        if phoneme[i] in oneWordByVowel:
            kana.append(dic[phoneme[i]])
            kanaDuration.append(duration[i])
            i+=1
        else:
            ph = phoneme[i]+phoneme[i+1]
            if ph in twoWordByTwoConsonant:
                kn = dic[ph]
                assert len(kn)==2
                kana+=[kn[0],kn[1]]
                kanaDuration+=[duration[i],duration[i+1]]
            else:
                kn=dic[ph]
                assert len(kn)==1
                kana+=[kn]
                kanaDuration+=[duration[i]+duration[i+1]]
            i+=2
    assert len(kana)==len(kanaDuration)
    return kana,kanaDuration



#きゃ、きゅ、きょなどを一つの平仮名として扱う
def Phoneme2Kana_ver1(phoneme,duration):
    kana=[]
    kanaDuration=[]
    N=len(phoneme)
    i=0
    while i<N:
        if phoneme[i] in oneWordByVowel:
            kana.append(dic[phoneme[i]])
            kanaDuration.append(duration[i])
            i+=1
        else:
            kana.append(dic[phoneme[i]+phoneme[i+1]])
            kanaDuration.append(duration[i]+duration[i+1])
            i+=2
    assert len(kana)==len(kanaDuration)
    return kana,kanaDuration


#推論時に使う
def Phoneme2Kana_inference(phoneme):
    kana=[]
    N=len(phoneme)
    i=0
    while i<N:
        if phoneme[i] in oneWordByVowel:
            kana.append(dic[phoneme[i]])
            i+=1
        else:
            ph = phoneme[i]+phoneme[i+1]
            if ph in twoWordByTwoConsonant:
                kn = dic[ph]
                assert len(kn)==2
                kana+=[kn[0],kn[1]]
            else:
                kn=dic[ph]
                assert len(kn)==1
                kana+=[kn]
            i+=2
    return kana


#強調文字が入っている時に音素からカナ文字を生成する
def Phoneme2Kana_emp(phoneme):
    phoneme=[openjtalk2julius(p) for p in phoneme]
    kana=[]
    N=len(phoneme)
    i=0
    while i<N:
        if phoneme[i]=='＊':
            kana+=['＊']
            i+=1
        elif phoneme[i] in oneWordByVowel:
            kana.append(dic[phoneme[i]])
            i+=1
        else:
            ph = phoneme[i]+phoneme[i+1]
            if ph in twoWordByTwoConsonant:
                kn = dic[ph]
                assert len(kn)==2
                kana+=[kn[0],kn[1]]
            else:
                kn=dic[ph]
                assert len(kn)==1
                kana+=[kn]
            i+=2
    return kana

def openjtalk2julius(p3):
    if p3 in ['A','I','U',"E", "O"]:
        return p3.lower()
    if p3 == 'cl':
        return 'q'
    if p3 == 'pau':
        return 'sp'
    return p3


#音素列から強調している区間を見つける
def get_emp_index(basename,speaker):
    #音素列を取得
    with open("phoneme/"+speaker+"/Emp/"+basename+".lab","r") as f:
        f=f.read()
        phoneme=f.split(" ")
    for i in range(len(phoneme)):
        phoneme[i]=openjtalk2julius(phoneme[i])
    phoneme=["sil"]+phoneme+["sil"]

    for i in range(len(phoneme)):
        if phoneme[i]=="＊":
            left=i
            break
    for i in range(len(phoneme)-1,-1,-1):
        if phoneme[i]=="＊":
            right=i-2
            break
    assert left!=-1 and right!=-1 and left<right
    return left,right

#音素列から強調している区間を見つける
def get_emp_index_en(basename):
    #音素列を取得 /home/sarulab/yoshifumi_nakano/GraduationThensis/raw_data/JECS_EN/JECS_EN
    with open("raw_data/JECS_EN/JECS_EN/"+basename+".lab","r") as f:
        f=f.read()
        phoneme=list(f)[1:]
    for i in range(len(phoneme)):
        phoneme[i]=comma2space(phoneme[i].lower())
    phoneme=["sil"]+phoneme+["sil"]

    phoneme_=[]
    for i in range(1,len(phoneme)):
        now=phoneme[i]
        past=phoneme[i-1]
        if now=="sp" and past==",":
            continue
        if now=="sp" and past==".":
            continue
        if now=="～" and past=="sp":
            continue

        phoneme_.append(now)
        
        

    phoneme=phoneme_
    phoneme=["sil"]+phoneme

    left=-1
    right=-1
    for i in range(len(phoneme)):
        if phoneme[i]=="*":
            left=i
            break
    for i in range(len(phoneme)-1,-1,-1):
        if phoneme[i]=="*":
            right=i-2
            break
    assert left!=-1 and right!=-1 and left<right
    
    p=[]
    for ph in phoneme:
        if ph!="*":
            p.append(ph)
    return left,right,p




def openjtalk2julius(p3):
    if p3 in ['A','I','U',"E", "O"]:
        return p3.lower()
    if p3 == 'cl':
        return 'q'
    if p3 == 'pau':
        return 'sp'
    return p3

def comma2space(p3):
    if p3 in [" ","","　"]:
        return "sp"
    return p3

paths=["000","001","002","003","004","005","006","007","008","009"]
for path in paths:
    path_="JE_EMPH"+path+"_EN"
    print(get_emp_index_en(path_))


# phoneme=['ny', 'i', 'u', 'n', 'i', 'y', 'o', 'N', 'k', 'a', 'i', 'sp', 'f', 'u', 'r', 'a', 'N', 's', 'u', 'n', 'o', 'j', 'u', 'gy', 'o', 'o', 'g', 'a', 'a', 'r', 'i', 'm', 'a', 's', 'u'] 
# duration=[17, 5, 3, 6, 4, 6, 7, 5, 7, 11, 5, 3, 8, 3, 5, 5, 3, 12, 3, 3, 4, 10, 2, 8, 5, 6, 6, 2, 8, 4, 4, 7, 6, 17, 2]
# print(Phoneme2Kana_ver2(phoneme,duration))
# paths=os.listdir("phoneme/JECS/Emp")
# for path in paths:
#     get_emp_index(path[:-4],"JECS")
#     with open("phoneme/JECS/Emp/"+path,"r") as f:
#         f=f.read()
#         phoneme=f.split(" ")
#         kana=Phoneme2Kana_emp(phoneme)
#         print(kana)

