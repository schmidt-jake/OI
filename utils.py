import pandas as pd
import numpy as np

def get_patients(hl_thresh=15):

	lcrc, bbdc = pd.read_excel(
	  'data/patients.xlsx',
	  sheet_name=['Aud_LCRC','Aud_BBDC'],
	  na_values=['','.','<No Form>','Unknown','Unknown or Not Reported']
	).values()

	lcrc['R Air PTA'] = lcrc[['Air Conduction RightEar500','Air Conduction RightEar1K','Air Conduction RightEar2K']].mean(1)
	lcrc['L Air PTA'] = lcrc[['AirConductionLeftEar500','AirConductionLeftEar1K','AirConductionLeftEar2K']].mean(1)
	lcrc['R Bone PTA'] = lcrc[['Bone Conduction RightEar500','Bone Conduction RightEar1K','Bone Conduction RightEar2K']].mean(1)
	lcrc['L Bone PTA'] = lcrc[['Bone Conduction LeftEar500','Bone Conduction LeftEar1K','Bone Conduction LeftEar2K']].mean(1)

	bbdc['R Air PTA'] = bbdc[['AirConductionRightEar500Hz','AirConductionRightEar1K','AirConductionRightEar2K']].mean(1)
	bbdc['L Air PTA'] = bbdc[['AirConductionLeftEar500','AirConductionLeftEar1K','AirConductionLeftEar2K']].mean(1)
	bbdc['R Bone PTA'] = bbdc[['BoneCondRight500','BoneCondRight1K','BoneCondRight2K']].mean(1)
	bbdc['L Bone PTA'] = bbdc[['BoneCondLeft500','BoneCondLeft1K','BoneCondLeft2K']].mean(1)
	bbdc = bbdc.rename(columns={'DateOfVisit':'VisitDate'})

	# combine data sets
	patients = lcrc.append(bbdc, sort=True).sort_values('VisitDate')
	patients['Age'] = (patients.VisitDate - patients.DOB).dt.days/365

	patients = patients[['LCRC ID','BBDC ID','Age','Gender','VisitDate','DOB','Subtype of OI','L Air PTA','L Bone PTA','R Air PTA','R Bone PTA']]

	# necessary for grouping by ID
	patients['LCRC ID'] = patients['LCRC ID'].fillna(-1).astype(int)
	patients['BBDC ID'] = patients['BBDC ID'].fillna(-1).astype(int)
	# patients = patients.groupby(['LCRC ID','BBDC ID']).ffill().drop_duplicates(['LCRC ID','BBDC ID'], keep='last')

	dobs = pd.read_excel('data/bbdc_no_dob.xlsx').rename(columns={'LCRC_ID':'LCRC ID'})
	dobs['LCRC ID'] = dobs['LCRC ID'].fillna(-1).astype(int)
	dobs['BBDC ID'] = dobs['BBDC ID'].fillna(-1).astype(int)

	patients = patients.merge(dobs[['LCRC ID','BBDC ID','Age At Visit']], how='left', on=['LCRC ID','BBDC ID'])
	patients['Age'] = patients['Age'].where(patients['Age'].notnull(), patients['Age At Visit'])

	# drop OI subtypes II and II/III and null subtype
	patients = patients[~patients['Subtype of OI'].isin([np.nan,'II','II/III'])]

	# patients = patients.dropna(subset=['L Air PTA','L Bone PTA','R Air PTA','R Bone PTA'],how='all')

	# drop records without any PTA
	patients.dropna(subset=['R Air PTA','L Air PTA'], how='all', inplace=True)
	# patients.dropna(subset=['R Bone PTA','L Bone PTA'], how='all', inplace=True) # ask Machol about this

	def filter_records(patient):
		completest_records = patient.loc[patient.notnull().sum(1) == patient.notnull().sum(1).max()]
		return completest_records.loc[[completest_records['VisitDate'].idxmax()]]

	patients = patients.groupby(['LCRC ID','BBDC ID']).apply(filter_records).reset_index(drop=True)

	# add features
	patients['consortium'] = np.where(patients['BBDC ID'] != -1,'BBDC','LCRC') # use BBDC over LCRC if patient was in both
	patients['consortium_ID'] = np.where(patients['BBDC ID'] != -1,patients['BBDC ID'],patients['LCRC ID'])
	patients['HL_sidedness'] = patients[['L Air PTA','R Air PTA']].gt(hl_thresh).sum(1).map(lambda n_HL_ears: ['none','Unilateral','Bilateral'][n_HL_ears])
	patients['age_bin'] = patients['Age'].floordiv(10).mul(10).clip(upper=60).fillna(-1).astype(int) # we chose ">60" as highest bucket

	# assign UIDs
	patients = patients.reset_index(drop=True)
	patients['UID'] = patients.index

	# NOTE: Negative PTA == healthy
	return patients

def patients2ears(patients, hl_thresh):
	# split into ear-level data
	ears = patients.melt(
	  id_vars=['UID','Gender','VisitDate','Subtype of OI','Age','age_bin','consortium','consortium_ID'],
	  value_vars=['L Air PTA','R Air PTA','L Bone PTA','R Bone PTA']
	)
	ears['side'] = np.where(ears['variable'].str.contains('^L\s'),'Left','Right')
	ears.variable = ears['variable'].str.lower().str.replace('^l\s|^r\s|\s','')
	ears.set_index([c for c in ears.columns if c != 'value'], inplace=True)
	ears = ears['value'].unstack('variable').reset_index()

	# ensure airbonegap is calculated correctly
	ears.eval('airbonegap = airpta - bonepta', inplace=True)

	# ensure there aren't instances where we could impute one value from the other two
	assert ears[['airpta','bonepta','airbonegap']].count(1).ne(2).all()

	# drop ears with any missing data
	# IMPORTANT: only 195 ears have nonnull airbonegap, and only 383 have nonnull bonepta (all 628 have nonnull airpta)

	ears.loc[ears.eval('airpta > @hl_thresh and bonepta <= @hl_thresh'),'HL_type'] = 'CHL'
	ears.loc[ears.eval('airpta > @hl_thresh and bonepta > @hl_thresh and airbonegap < 15'),'HL_type'] = 'SNHL'
	ears.loc[ears.eval('airpta > @hl_thresh and bonepta > @hl_thresh and airbonegap >= 15'),'HL_type'] = 'MHL'

	# add ear-level features
	ears.eval('''
	  CHL = HL_type == "CHL"
	  SNHL = HL_type == "SNHL"
	  MHL = HL_type == "MHL"
	''',inplace=True)

	ears.loc[ears.eval('airpta > 20 and airpta <= 40'),'severity'] = 'mild'
	ears.loc[ears.eval('airpta > 40 and airpta <= 70'),'severity'] = 'moderate'
	ears.loc[ears.eval('airpta > 70 and airpta <= 90'),'severity'] = 'severe'
	ears.loc[ears.eval('airpta > 90'),'severity'] = 'profound'

	# ensure the HL types are mutually exclusive:
	assert ears[['CHL','SNHL','MHL']].sum(1).max() == 1, 'HL type not mutually exclusive!'
	
	return ears