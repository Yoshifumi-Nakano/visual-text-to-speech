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
                kanaDuration+=[duration[i]+duration[i+1]]
            i+=2
    return kana

# phoneme=['ny', 'i', 'u', 'n', 'i', 'y', 'o', 'N', 'k', 'a', 'i', 'sp', 'f', 'u', 'r', 'a', 'N', 's', 'u', 'n', 'o', 'j', 'u', 'gy', 'o', 'o', 'g', 'a', 'a', 'r', 'i', 'm', 'a', 's', 'u'] 
# duration=[17, 5, 3, 6, 4, 6, 7, 5, 7, 11, 5, 3, 8, 3, 5, 5, 3, 12, 3, 3, 4, 10, 2, 8, 5, 6, 6, 2, 8, 4, 4, 7, 6, 17, 2]
# print(Phoneme2Kana_ver2(phoneme,duration))
        
