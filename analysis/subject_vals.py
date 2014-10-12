'''
Created on Aug 17, 2014

@author: andrew
'''

# M100 window:
m100_time = {
             'R0053': 0.125, #b,r
             'R0411': 0.125, #b,r
             'R0606': '????',
             'R0684': 0.130, #b,r
             'R0687': 0.115, #b,r
             'R0703': 0.130,
             'R0734': 0.120,
             'R0737': '????',
             'R0749': '????',
             'R0816': 0.125, #b,r
             'R0823': 0.130, #b,r
             'R0845': 0.135, #b,r
             'R0850': 0.135, #b,r
             'R0854': 0.125, #b,r
             'R0855': 0.115, #b,r
             'R0857': '????',
             }
m100_sensor = {
               'R0053': 'MEG 026',
               'R0411': 'MEG 063',
               'R0606': '????',
               'R0684': 'MEG 015',
               'R0687': 'MEG 014',
               'R0703': 'MEG 058',
               'R0734': 'MEG 015',
               'R0737': '????',
               'R0749': '????',
               'R0816': 'MEG 015',
               'R0823': 'MEG 092',
               'R0845': 'MEG 015',
               'R0850': 'MEG 073',
               'R0854': 'MEG 014',
               'R0855': 'MEG 026',
               'R0857': '????'
               }
# M170 window:
m170_max = {
             'R0053': 0.185, #r,b
             'R0411': 0.180, #r,b
             'R0606': 0.160, #r,b
             'R0684': 0.180, #r,b
             'R0687': 0.180, #r,b
             'R0703': 0.180, #r,b
             'R0734': 0.180, #r,b
             'R0737': 0.160, #r,b
             'R0749': 0.160, #r,b
             'R0816': 0.175, #r,b
             'R0823': 0.185, #r,b
             'R0845': 0.175, #r,b
             'R0850': 0.185, #r,b
             'R0854': 0.175, #r,b
             'R0855': 0.170, #r,b
             'R0857': 0.180, #r,b
             }
m170_time = {x: (round(m170_max[x]-0.02,3), round(m170_max[x]+0.02,3)) for x in m170_max}
m170_sensor = {
               'R0053': 'MEG 059',
               'R0411': 'MEG 063',
               'R0606': 'MEG 015',
               'R0684': 'MEG 015',
               'R0687': 'MEG 014',
               'R0703': 'MEG 026',
               'R0734': 'MEG 014',
               'R0737': 'MEG 015',
               'R0749': 'MEG 015',
               'R0816': 'MEG 026',
               'R0823': 'MEG 042',
               'R0845': 'MEG 015',
               'R0850': 'MEG 073',
               'R0854': 'MEG 014',
               'R0855': 'MEG 060',
               'R0857': 'MEG 042',
               }

 
replaceNames_old = {
'fre.P|@*':'plr.lem',
'fre.S|@*':'sng.lem',
'fre.4|@*':'verb.lem',
'fre.i|@':'verb.stm',
'fre.1|@*DEVRAT':'noun.lem.DEVRAT',
'fre.P|@+s':'noun.s',
'fre.log(@+s)':'log.s',
'fre.S|@-P|@+s':'DELTA',
'fre.@':'stm',
'fre.log(@)':'log.stm',
'fre.1|@*':'noun.lem',
'fre.@+ed':'ed',
'fre.i|@*':'verbInfinitive.lem',
'fre.pe|@*':'ing.lem',
'fre.e3S|@*':'verbPresent.lem',
'fre.log(@+ed)':'log.ed',
'fre.@*':'lem',
'fre.log(@*)':'log.lem',
'fre.@+s':'s',
'fre.4|@*DEVRAT':'verb.lem.DEVRAT',
'fre.S|@':'noun.stm',
'fre.a|@*':'past.lem',
'fre.e3S|@+s':'verb.s',
'clx.1.MorphNum':'stm.MorphNum',
'clx.1.LetNum':'stm.LetNum',
'clx.1.SylNum':'stm.SylNum',
'clx.1.deriv_family_size':'lem.derFamSize',
'clx.1.log(deriv_family_size)':'log.lem.derFamSize',
'clx.P.LetNum':'s.LetNum',
'clx.P.SylNum':'s.SylNum',
'clx.a1S.LetNum':'ed.LetNum',
'clx.a1S.SylNum':'ed.SylNum',
'SUBTLWF':'slx.stm',
'LgSUBTLWF':'log.slx.stm',
'SUBTLCD':'slxCD.stm',
'LgSUBTLCD':'log.slxCD.stm',
'OLD':'OLD.stm',
'OLDF':'OLDF.stm',
'PLD':'PLD.stm',
'PLDF':'PLDF.stm',
'slx.PaFre':'slx.ed',
'slx.log(PaFre)':'log.slx.ed',
'slx.AdjFre':'slx.adj',
'slx.Vfre':'slx.verb',
'fre.4|@+ed':'verb.ed',
'fre.2|@+ed':'adj.ed',
'wrd':'Fre',
'log.wrd':'LogFre',
'verb.wrd':'VerbGivenWord',
'wrd.LetNum':'LetNum',
'wrd.SylNum':'SylNum',
'wrd.MorphNum':'MorphNum',
'wrd.UN2_F':'UnconstrainedBigramFre',
'wrd.Orth':'OrthNeighborhoodSize',
'wrd.Orth_F':'OrthNeighborhoodFre',
'stm.dom':'Ambiguity_Root',
'wrd.dom':'Ambiguity',
'lem.dom':'Ambiguity_Lemma',
's.dom':'Ambiguity_RootS',
'ed.dom':'Ambiguity_RootEd',
'AfxSurprisal':'ParadigmSurprisal'
}

replaceNames = {
'fre.P|@*':'PluralNounGivenLemma',
'fre.S|@*':'SingularNounGivenLemma',
'fre.4|@*':'VerbGivenLemma',
'fre.i|@':'VerbGivenStem',
'fre.1|@*DEVRAT':'DEVRAT_NounGivenLemma',
'fre.P|@+s':'PluralGivenStemS',
'fre.log(@+s)':'LogFre_StemS',
'fre.S|@-P|@+s':'DELTA',
'fre.@':'Fre_Stem',
'fre.log(@)':'LogFre_Stem',
'fre.1|@*':'NounGivenLemma',
'fre.@+ed':'StemEd',
'fre.i|@*':'InfinitiveVerbGivenLemma',
'fre.pe|@*':'PresentParticipleGivenLemma',
'fre.e3S|@*':'e3SVerbGivenLemma',
'fre.log(@+ed)':'LogFre_StemEd',
'fre.@*':'Fre_Lemma',
'fre.log(@*)':'LogFre_Lemma',
'fre.@+s':'Fre_StemS',
'fre.4|@*DEVRAT':'DEVRAT_VerbGivenLemma',
'fre.S|@':'SingularNounGivenStem',
'fre.a|@*':'StemEdGivenLemma',
'fre.e3S|@+s':'e3SVerbGivenStemS',
'clx.1.MorphNum':'MorphNum_Stem',
'clx.1.LetNum':'LetNum_Stem',
'clx.1.SylNum':'SylNum_Stem',
'clx.1.deriv_family_size':'DerivationalFamilySize',
'clx.1.log(deriv_family_size)':'Log_DerivationalFamilySize',
'clx.P.LetNum':'LetNum_StemS',
'clx.P.SylNum':'SylNum_StemS',
'clx.a1S.LetNum':'LetNum_StemEd',
'clx.a1S.SylNum':'SylNum_StemEd',
'wn.SynsetCnt': 'SynsetNum_Stem',
'wn.log(SynsetCnt)': 'LogSynsetNum_Stem',
'SUBTLWF':'SUBTLEXFre_Stem',
'LgSUBTLWF':'LogSUBTLEXFre_Stem',
'SUBTLCD':'ContextualDiversity_Stem',
'LgSUBTLCD':'LogContextualDiversity_Stem',
'OLD':'OLD_Stem',
'OLDF':'OLDF_Stem',
'PLD':'PLD_Stem',
'PLDF':'PLDF_Stem',
'slx.PaFre':'SUBTLEXFre_StemEd',
'slx.log(PaFre)':'LogSUBTLEXFre_StemEd',
'slx.AdjFre':'SUBTLEXFre_PastParticipleStemEd',
'slx.Vfre':'SUBTLEXFre_PastTenseStemEd',
'fre.4|@+ed':'PastTenseGivenStemEd',
'fre.2|@+ed':'PastParticipleStemEd',
'wrd':'Fre',
'log.wrd':'LogFre',
'verb.wrd':'VerbGivenWord',
'wrd.LetNum':'LetNum',
'wrd.SylNum':'SylNum',
'wrd.MorphNum':'MorphNum',
'wrd.UN2_F':'UnconstrainedBigramFre',
'wrd.Orth':'OrthNeighborhoodSize',
'wrd.Orth_F':'OrthNeighborhoodFre',
'stm.dom':'Ambiguity_Stem',
'wrd.dom':'Ambiguity',
'lem.dom':'Ambiguity_Lemma',
's.dom':'Ambiguity_StemS',
'ed.dom':'Ambiguity_StemEd',
'AfxSurprisal':'ParadigmSurprisal'
}