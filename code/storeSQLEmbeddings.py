from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

few_shots = {
"Give me spectroscopically-classified supernovae within z<0.015.":"SELECT t.*, h.* FROM YSE_App_transient t INNER JOIN YSE_App_host h ON h.id = t.host_id WHERE t.host_id IS NOT NULL AND (t.redshift OR h.redshift) IS NOT NULL AND COALESCE(t.redshift, h.redshift) < 0.015 AND (t.TNS_spec_class like '%SN%');",

"Find me all fast and young supernovae.":"SELECT DISTINCT t.name, t.TNS_spec_class AS `classification`, g.first_detection AS `first_detection`, g.latest_detection AS `latest_detection`, g.number_of_detection AS `number_of_detection`, og.name AS `group_name` FROM (SELECT DISTINCT	t.id, MIN(pd.obs_date) AS `first_detection`, MAX(pd.obs_date) AS `latest_detection`, COUNT(pd.obs_date) AS `number_of_detection` FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id WHERE (pd.flux/pd.flux_err > 2 OR pd.mag_err< 0.2 OR ((pd.mag_err IS NULL) AND (pd.mag IS NOT NULL))) GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id WHERE t.TNS_spec_class IS NULL AND t.name NOT LIKE '%YSEAGN%' AND TO_DAYS(CURDATE())- TO_DAYS(first_detection) < 6 AND TO_DAYS(CURDATE())- TO_DAYS(latest_detection) < 3 AND number_of_detection > 1 ORDER BY g.first_detection DESC;",

"Give me the properties of all supernovae spectroscopically classified as a Type SN Ibn.": "SELECT t.* FROM YSE_App_transient t WHERE (t.TNS_spec_class LIKE 'SN Ibn');",

"Give me the names of all supernovae spectroscopically classified as a Type SN Ic.": "SELECT t.name FROM YSE_App_transient t WHERE (t.TNS_spec_class LIKE 'SN Ic');",

"Give me all transients in the database with associated host galaxy information.":"SELECT t.* FROM YSE_App_transient t INNER JOIN YSE_App_host h ON h.id = t.host_id;",

"Of all transients are current tagged as 'Young', which one was discovered most recently?":"SELECT t.name, tg.name, t.disc_date as `tag` FROM YSE_App_transient t INNER JOIN YSE_App_transient_tags AS tt ON t.id = tt.transient_id INNER JOIN YSE_App_transienttag AS tg ON tt.transienttag_id = tg.id WHERE tg.name = 'Young' ORDER BY t.disc_date DESC LIMIT 1;",

"Give me the names of all galaxies matched to named supernovae in 2023.":"SELECT h.name , t.name FROM YSE_App_host h INNER JOIN YSE_App_transient h ON t.host_id=h.id WHERE t.name LIKE '2023%';",

"Give me the names, coordinates, and spectroscopic class of all Southern-hemisphere (dec<-30 deg) transients.":"SELECT DISTINCT t.name, t.ra, t.dec, t.TNS_spec_class FROM YSE_App_transient t WHERE t.dec <= -30  AND (DATEDIFF(curdate(), t.disc_date) <= 5) ;",

"How many spectroscopically-confirmed SNe Ia are in the database?":"SELECT COUNT(DISTINCT t.name) FROM YSE_App_transient t WHERE t.TNS_spec_class LIKE 'SN Ia';",

"Which supernovae are spectroscopically-confirmed?":"SELECT t.* FROM YSE_App_transient t WHERE t.TNS_spec_class LIKE 'SN%';",

"How far away is SN 2023bee?":"SELECT t.redshift FROM YSE_App_transient t WHERE t.name LIKE '2023bee';",

"When did supernova 2022acjh have its max brightness (in phase and MJD/date) and in what filter?":"SELECT DISTINCT pd.obs_date AS `observed_date`, TO_DAYS(pd.obs_date)-TO_DAYS(t.disc_date), t.disc_date as `discovery_date`, pb.name AS `filter`, pd.mag, pd.mag_err,t.mw_ebv FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id INNER JOIN YSE_App_photometricband pb ON pb.id = pd.band_id WHERE t.name LIKE '2022acjh' AND pd.mag IS NOT NULL AND pd.mag_err < 0.3 ORDER BY pd.mag ASC LIMIT 1;",

"Which supernova is currently brightest between SN 2023zcu, SN 2023zzk, and SN 2023xvj?":"SELECT DISTINCT t.name, g.latest_detection AS `latest_detection`, pd.* FROM (SELECT DISTINCT t.id,MAX(pd.obs_date) AS `latest_detection` FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id WHERE (pd.mag IS NOT NULL) GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id WHERE (TO_DAYS(CURDATE())- TO_DAYS(latest_detection)) < 10 AND (t.NAME LIKE '2023zcu' OR t.NAME LIKE '2023zzk' OR t.NAME LIKE '2023xvj')  AND pd.obs_date = g.latest_detection ORDER BY pd.mag ASC LIMIT 1;",

"Of the transients overlapping with HST, which was brightest in the last 10 days?":"SELECT DISTINCT t.name, g.latest_detection AS `latest_detection`, t.has_hst, pd.* FROM (SELECT DISTINCT t.id,MAX(pd.obs_date) AS `latest_detection` FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id WHERE (pd.mag IS NOT NULL) GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id WHERE ((TO_DAYS(CURDATE())- TO_DAYS(latest_detection)) < 10) AND (pd.obs_date = g.latest_detection) AND (t.has_hst > 0) ORDER BY pd.mag ASC LIMIT 1;",

"Of all the transients with a `TESS` tag, which is currently the brightest?":"SELECT DISTINCT t.name, g.latest_detection AS `latest_detection`, tg.name as `tag`, pd.* FROM (SELECT DISTINCT t.id,MAX(pd.obs_date) AS `latest_detection` FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id WHERE (pd.mag IS NOT NULL) GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id JOIN YSE_App_transient_tags AS tt ON t.id = tt.transient_id JOIN YSE_App_transienttag AS tg ON tt.transienttag_id = tg.id WHERE tg.name = 'TESS' AND ((TO_DAYS(CURDATE())- TO_DAYS(latest_detection)) < 3) AND (pd.obs_date = g.latest_detection) ORDER BY pd.mag ASC LIMIT 1;",

"Of the transients overlapping with HST, which is currently brightest?":"SELECT DISTINCT t.name, g.latest_detection AS `latest_detection`, t.has_hst, pd.* FROM (SELECT DISTINCT t.id,MAX(pd.obs_date) AS `latest_detection` FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id WHERE (pd.mag IS NOT NULL) GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id WHERE ((TO_DAYS(CURDATE())- TO_DAYS(latest_detection)) < 3) AND (pd.obs_date = g.latest_detection) AND (t.has_hst > 0) ORDER BY pd.mag ASC LIMIT 1;",

"How many SNe II have been discovered in 2023?":"SELECT COUNT(DISTINCT t.name) FROM YSE_App_transient t WHERE (t.name LIKE '2023%') AND (t.TNS_spec_class LIKE 'SN II')",

"Give me the names and coordinates of all transients between 20<dec<80 degrees.":"SELECT DISTINCT t.name, t.ra, t.dec FROM YSE_App_transient t WHERE t.dec >20 AND t.dec < 80;",

"Show me all transients with HST observations.":"SELECT t.name, t.ra, t.`dec`, t.disc_date, t.redshift, t.TNS_spec_class, t.has_hst FROM YSE_App_transient t WHERE t.has_hst > 0;",

"How many transients did YSE discover in the year 2021?":"SELECT COUNT(DISTINCT t.name) FROM YSE_App_transient t INNER JOIN YSE_App_transient_tags tt ON tt.transient_id = t.id INNER JOIN YSE_App_transienttag tg ON tg.id = tt.transienttag_id WHERE (tg.name = 'YSE' OR tg.name = 'YSE Forced Phot') AND (t.name LIKE '2021%');",

"How many transients are in the database from the year 2023?":"SELECT COUNT(DISTINCT t.name) FROM YSE_App_transient t WHERE (t.name LIKE '2023%')",

"Give me the properties of all transients with one of the following statuses:`Interesting/Watch/Following/FollowupRequested/New`":"SELECT t.* FROM YSE_App_transient t INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (ts.name = 'New' OR ts.name = 'Interesting' OR ts.name = 'Watch' OR ts.name = 'Following' OR ts.name = 'FollowupRequested');",

"How many supernovae have a status of 'Interesting'?":"SELECT COUNT(DISTINCT t.name) FROM YSE_App_transient t INNER JOIN YSE_App_transientstatus ts on ts.id=t.status_id WHERE ts.name LIKE 'Interesting';",

"What supernovae were given a status of `Interesting` in 2021?":"SELECT t.name FROM YSE_App_transient t INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (ts.name = 'Interesting') AND (t.name LIKE '2021%');",

"What transients currently have follow-up requested?":"SELECT t.name FROM YSE_App_transient t INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (ts.name = 'FollowupRequested');",

"What is the current status of SN 2023mjo?":"SELECT ts.name FROM YSE_App_transient t INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (t.name LIKE '2023mjo');",

"Get the statuses of all transients.":"SELECT DISTINCT t.name as `transient_name`, m.name as `status` FROM YSE_App_transientstatus t INNER JOIN YSE_App_transient m on t.id = m.status_id;",

"What transients currently set to 'Watch' status?":"SELECT t.name FROM YSE_App_transient t INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (ts.name = 'Watch');",

"Return the properties of all stripped envelope supernovae (SESNe) discovered in the last 2 years.":"SELECT t.* FROM YSE_App_transient t AND DATEDIFF(curdate(),t.disc_date) < 730 AND t.TNS_spec_class IN ('SN Ic', 'SN Ibc', 'SN Ib', 'SN IIb');",

"Get all photometry for the transient SN 2019ehk.":"SELECT DISTINCT pd.obs_date AS `observed_date`, TO_DAYS(pd.obs_date) AS `count_date`, pb.name AS `filter`, pd.mag, pd.mag_err,t.mw_ebv, pd.forced FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id INNER JOIN YSE_App_photometricband pb ON pb.id = pd.band_id WHERE t.name LIKE '2019ehk';",

"Which supernovae have photometry in the database?":"SELECT DISTINCT t.name FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id;",

"What are the spectroscopic classes of all supernovae in the database?":"SELECT t.name,t.TNS_spec_class FROM YSE_App_transient;",

"How many type Ia supernovae within z<0.1 have HST observations?" : "SELECT t.name, t.ra AS transient_RA, t.TNS_spec_class, t.`dec` AS transient_Dec, t.disc_date AS disc_date, t.redshift, t.TNS_spec_class, t.has_hst FROM YSE_App_transient t WHERE t.redshift < 0.02 AND t.has_hst > 0 AND t.TNS_spec_class LIKE 'SN Ia';",

"Get me all supernovae brighter than 17th mag and observed less than 7 days ago.":"SELECT DISTINCT t.* pb.name AS `filter`, pd.mag AS 'Recent mag', pd.obs_date FROM YSE_App_transientphotdata pd INNER JOIN YSE_App_transientphotometry tp ON pd.photometry_id = tp.id INNER JOIN YSE_App_transient t ON tp.transient_id = t.id INNER JOIN YSE_App_photometricband pb ON pb.id = pd.band_id WHERE pd.mag < 17 AND DATEDIFF(curdate(), pd.obs_date) < 7;",

"Get me the names of all supernovae observed by TESS.":"SELECT DISTINCT t.name FROM YSE_App_transient t INNER JOIN YSE_App_transient_tags tt ON tt.transient_id = t.id INNER JOIN YSE_App_transienttag tg ON tg.id = tt.transienttag_id WHERE tg.name = 'TESS';",

"What is the Milky Way extinction of 2023bee?":"SELECT t.mw_ebv from YSE_App_transient t where t.name like '2023bee';",

"Give me the names and coordinates of all transients in low-extinction regions.":"SELECT t.name, t.ra, t.dec, t.mw_ebv FROM YSE_App_transient t WHERE t.mw_ebv < 0.3;",

"What is the photometric classification of SN 2020oi?":"SELECT t.name as `transient_name`, tc.name as `photometric_classification` FROM YSE_App_transient t INNER JOIN YSE_App_transientclass tc on tc.id = t.photo_class_id WHERE t.name LIKE '2020oi';",

"How many SNe II are there, classified photometrically or spectroscopically?":"SELECT COUNT(DISTINCT t.name) FROM YSE_App_transient t INNER JOIN YSE_App_transientclass tc on tc.id = t.photo_class_id WHERE (tc.name LIKE 'SN II') OR (t.TNS_spec_class LIKE 'SN II');",

"Give me the names of all transients within z < 0.5 where the spectroscopic and photometric classifications disagree.":"SELECT DISTINCT t.name, tc.name as `photometric_classification`, t.TNS_spec_class as `spectroscopic_classification` FROM YSE_App_transient t INNER JOIN YSE_App_transientclass tc on tc.id = t.photo_class_id WHERE (tc.name != t.TNS_spec_class) AND tc.name IS NOT NULL AND t.TNS_spec_class IS NOT NULL AND t.redshift < 0.5;",

"Which supernovae have disagreeing spectroscopic and photometric classifications?":"SELECT DISTINCT t.name, tc.name, t.TNS_spec_class FROM YSE_App_transient t INNER JOIN YSE_App_transientclass tc on tc.id = t.photo_class_id WHERE (tc.name != t.TNS_spec_class) AND tc.name IS NOT NULL AND t.TNS_spec_class IS NOT NULL;",

"What supernovae are spectroscopically-confirmed beyond z>0.5?":"SELECT DISTINCT t.name, t.TNS_spec_class, t.redshift WHERE t.redshift>0.5 AND t.TNS_spec_class LIKE 'SN%';",

"Which confirmed SNe Ia have a photometric classification that is not SN Ia?":"SELECT DISTINCT t.name, tc.name, t.TNS_spec_class FROM YSE_App_transient t INNER JOIN YSE_App_transientclass tc on tc.id = t.photo_class_id WHERE t.TNS_spec_class LIKE 'SN Ia' AND (tc.name NOT LIKE 'SN Ia') AND tc.name IS NOT NULL;",

"Get all transients that are <0.2 arcsec from their host galaxies and not spectroscopically classified.":"SELECT DISTINCT	t.name, t.TNS_spec_class AS `classification`, t.ra as `transient_ra`, t.dec as `transient_dec`, (ACOS(SIN(RADIANS(t.dec))*SIN(RADIANS(h.`dec`)) + COS(RADIANS(t.dec))*COS(RADIANS(h.`dec`))*COS(RADIANS(ABS(t.ra - h.ra)))))*206265 AS `separation` FROM YSE_App_transient t INNER JOIN YSE_App_host h ON h.id = t.host_id WHERE t.TNS_spec_class IS NULL AND (ACOS(SIN(RADIANS(t.dec))*SIN(RADIANS(h.`dec`)) + COS(RADIANS(t.dec))*COS(RADIANS(h.`dec`))*COS(RADIANS(ABS(t.ra - h.ra)))))*206265 < 0.2;",

"What transients had new photometry taken in the last week?":"SELECT DISTINCT t.name, t.TNS_spec_class AS `classification`, g.latest_detection AS `latest_detection` FROM (SELECT DISTINCT t.id,MAX(pd.obs_date) AS `latest_detection` FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id WHERE (pd.mag IS NOT NULL) GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id WHERE TO_DAYS(CURDATE())- TO_DAYS(latest_detection) < 7;",

"What transients have had new observations taken at any time in the last 2 days?":"SELECT DISTINCT t.name, t.TNS_spec_class AS `classification`, g.latest_detection AS `latest_detection` FROM (SELECT DISTINCT t.id,MAX(pd.obs_date) AS `latest_detection` FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id WHERE (pd.mag IS NOT NULL) GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id WHERE TO_DAYS(CURDATE())- TO_DAYS(latest_detection) < 2;",

"What supernovae have we observed in the last 3 days?":"SELECT DISTINCT t.name, t.TNS_spec_class AS `classification`, g.latest_detection AS `latest_detection` FROM (SELECT DISTINCT t.id,MAX(pd.obs_date) AS `latest_detection` FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id WHERE (pd.mag IS NOT NULL) GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id WHERE TO_DAYS(CURDATE())- TO_DAYS(latest_detection) < 3 AND t.TNS_spec_class LIKE 'SN%';",

"What is the current brightness of 2021baf?":"SELECT DISTINCT t.name, g.latest_detection AS `latest_detection`, pd.* FROM (SELECT DISTINCT t.id,MAX(pd.obs_date) AS `latest_detection` FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id WHERE (pd.mag IS NOT NULL) GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id WHERE (TO_DAYS(CURDATE())- TO_DAYS(latest_detection)) < 3 AND (t.NAME LIKE '2021baf') AND pd.obs_date = g.latest_detection ORDER BY pd.mag ASC LIMIT 1;",

"What spectroscopically-confirmed TDEs have we observed in the last 5 days?":"SELECT DISTINCT t.name, t.TNS_spec_class AS `classification`, g.latest_detection AS `latest_detection` FROM (SELECT DISTINCT t.id,MAX(pd.obs_date) AS `latest_detection` FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id WHERE (pd.mag IS NOT NULL) GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id WHERE TO_DAYS(CURDATE())- TO_DAYS(latest_detection) < 5 AND t.TNS_spec_class LIKE 'TDE';",

"Find me all transients that, at any point in the last week, have increased in brightness in any band by greater than 0.3 mag.":"SELECT D.name, D.ra, D.dec, D.class, D.obs_date, D.mag, D.mag_err, D.band, D.SecondDetection, D.mag2, D.DeltaMag, D.Rise FROM (SELECT * FROM (SELECT *,(B.mag2-B.mag) AS 'DeltaMag',(B.mag2-B.mag)/DATEDIFF(B.obs_date, B.obs_date2) AS 'Rise' FROM (SELECT * FROM  (SELECT pd.photometry_id,   pd.obs_date,   pd.mag,   pd.band_id, pd.id,  pd.mag_err, pd.flux, pd.flux_err,  pd.flux_zero_point, (SELECT MAX(pd2.obs_date) FROM YSE_App_transientphotdata pd2 WHERE ((DATEDIFF(CURDATE(), pd2.obs_date) < 7) AND (pd2.photometry_id = pd.photometry_id)  AND (pd2.id != pd.id) AND (pd2.band_id = pd.band_id) AND (DATEDIFF(pd.obs_date, pd2.obs_date) > 0)   AND (pd2.obs_date NOT IN (SELECT MAX(pdmax.obs_date) FROM YSE_App_transientphotdata pdmax  WHERE (pdmax.photometry_id = pd.photometry_id) AND (pdmax.band_id = pd.band_id))))) AS 'SecondDetection' FROM YSE_App_transientphotdata pd WHERE (pd.flux-3*pd.flux_err)>0) A INNER JOIN (SELECT pd3.obs_date AS 'obs_date2', pd3.photometry_id AS 'p_id2', IF((pd3.flux-3*pd3.flux_err)<0, -2.5*LOG10(3*pd3.flux_err)-27.5, pd3.mag) AS 'mag2', pd3.band_id AS 'band2', pd3.flux AS 'flux2',   pd3.flux_err AS 'fluxerr2', pd3.mag_err AS 'mag_err2' FROM YSE_App_transientphotdata pd3) obs2 ON (obs2.obs_date2=A.SecondDetection AND obs2.p_id2 = A.photometry_id  AND obs2.band2 = A.band_id) WHERE (DATEDIFF(CURDATE(), A.obs_date) < 4) AND ISNULL(A.SecondDetection) = FALSE) B) C INNER JOIN  (SELECT DISTINCT ty.id AS 'photID', ty.transient_id AS 'tid'  FROM YSE_App_transientphotometry ty) ts ON ts.photID=C.photometry_id  INNER JOIN  (SELECT pb.id AS 'b_id',  pb.name AS 'band' FROM YSE_App_photometricband pb) pb ON pb.b_id=C.band_id INNER JOIN (SELECT t.name,  t.ra,  t.dec,  t.id AS 'tid2',  t.TNS_spec_class AS 'class' FROM YSE_App_transient t) tt ON tt.tid2=ts.tid WHERE C.Rise >= 0.3) D WHERE D.mag_err < 0.35;",

"What was the last date that SN2023ixf was observed?":"SELECT DISTINCT t.name, g1.max_date FROM YSE_App_transientphotdata pd INNER JOIN YSE_App_transient t INNER JOIN (SELECT a3.id AS transient_id, MAX(obs_date) AS max_date FROM YSE_App_transientphotdata a1 INNER JOIN YSE_App_transientphotometry a2 ON a1.photometry_id = a2.id INNER JOIN YSE_App_transient a3 ON a2.transient_id = a3.id WHERE mag IS NOT NULL GROUP BY a3.id) AS g1 ON g1.transient_id = t.id WHERE t.name LIKE '2023ixf';",

"What transients have had new photometry taken in the last week?":"SELECT DISTINCT t.name, t.ra,t.dec, pd.obs_date AS `observed_date`, TO_DAYS(pd.obs_date) AS `count_date`, pb.name AS `filter`, pd.mag, pd.mag_err,t.mw_ebv, og.name AS `group_name`, pd.forced FROM YSE_App_transient t INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id INNER JOIN YSE_App_photometricband pb ON pb.id = pd.band_id WHERE t.disc_date IS NOT NULL AND TO_DAYS(CURDATE()) - TO_DAYS(pd.obs_date) < 7;",

"Get the properties of all transients discovered in the last 40 days.":"SELECT DISTINCT	t.name, t.ra,t.dec, pd.obs_date AS `observed_date`, TO_DAYS(pd.obs_date) AS `count_date`, pb.name AS `filter`, pd.mag, pd.mag_err,t.mw_ebv, og.name AS `group_name`, pd.forced FROM YSE_App_transient t INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id INNER JOIN YSE_App_photometricband pb ON pb.id = pd.band_id WHERE t.disc_date IS NOT NULL AND TO_DAYS(CURDATE())- TO_DAYS(t.disc_date) < 40 ORDER BY t.name ASC, TO_DAYS(pd.obs_date) DESC;",

"Give me the general properties of SN2022xxf.":"SELECT DISTINCT t.* FROM YSE_App_transient t INNER JOIN YSE_App_transient_tags tt ON tt.transient_id = t.id INNER JOIN YSE_App_transienttag tg ON tg.id = tt.transienttag_id INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (tg.name = 'YSE' OR tg.name = 'YSE Forced Phot') AND (t.name LIKE '2022xxf');",

"At what spectroscopic redshift did SN2020oi occur?":"SELECT t.name, t.redshift FROM YSE_App_transient WHERE (t.name LIKE '2020oi');",

"What is the host galaxy of SN 2020fqv?":"SELECT h.name, h.ra, h.`dec` FROM YSE_App_host h INNER JOIN YSE_App_transient t ON t.host_id = h.id WHERE t.name LIKE '2020fqv';",

"How many spectroscopically-confirmed SNe Ib are there within z < 0.1?": "SELECT COUNT(DISTINCT t.name) FROM YSE_App_transient t WHERE t.TNS_spec_class LIKE 'SN Ib' AND t.redshift < 0.1;",

"How many photometric SNe Ib are there within z < 0.1?": "SELECT COUNT(DISTINCT t.name) FROM YSE_App_transient t INNER JOIN YSE_App_transientclass tc on tc.id = t.photo_class_id WHERE tc.name LIKE 'SN Ib' AND t.redshift < 0.1;",

"Which SNe have photometric redshifts and not spectroscopic ones?": "SELECT t.name, h.photo_z, h.photo_z_err FROM YSE_App_transient t INNER JOIN YSE_App_host h on t.host_id = h.id WHERE t.redshift IS NULL AND h.photo_z IS NOT NULL;",

"Give me all spectroscopically-confirmed stripped-envelope supernovae.":"SELECT t.* FROM YSE_App_transient t WHERE t.TNS_spec_class IN ('SN Ic', 'SN Ibc', 'SN Ib', 'SN Ic-BL','SN Ibn','SN Icn','SN Ib-pec','SN Ib/c','SN Ic-pec')",

"How many spectra were taken for 2020fqv?": "SELECT COUNT(DISTINCT ts.obs_date) FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id WHERE t.name LIKE '2020fqv';",

"When were spectra taken for 2022oqm?": "SELECT DISTINCT ts.obs_date FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id WHERE t.name LIKE '2022oqm';",

"When were the spectra taken for SN 2023ixf?": "SELECT DISTINCT ts.obs_date FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id WHERE t.name LIKE '2023ixf';",

"What spectra were taken for 2023bee?":"SELECT DISTINCT t.name, ts.obs_date, i.* FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id INNER JOIN YSE_App_instrument i ON i.id = ts.instrument_id WHERE t.name LIKE '2023bee';",

"Get me the spectrum of 2020acct taken on Dec. 11th, 2020.":"SELECT DISTINCT tsd.wavelength, tsd.flux, tsd.wavelength_err, tsd.flux_err FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id INNER JOIN YSE_App_instrument i ON i.id = ts.instrument_id WHERE t.name LIKE '2020acct' AND ABS((TO_DAYS(ts.obs_date)-TO_DAYS('2020-12-11')) < 1);",

"Get the properties of the final spectrum taken for 2020acct.":"SELECT DISTINCT t.name, ts.*, i.* FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id INNER JOIN YSE_App_instrument i ON i.id = ts.instrument_id WHERE t.name LIKE '2020acct' ORDER BY ts.obs_date DESC LIMIT 1;",

"Get me the data from the last spectrum taken for 2020acct.":"SELECT DISTINCT tsd.wavelength, tsd.flux, tsd.wavelength_err, tsd.flux_err FROM (SELECT DISTINCT MAX(ts.obs_date) AS `last_spec_date`, t.id FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id INNER JOIN YSE_App_instrument i ON i.id = ts.instrument_id WHERE t.name LIKE '2020acct' AND ts.obs_date = g.last_spec_date;",

"How many days after SN 2022xxf's discovery was the last spectrum taken?":"SELECT DATEDIFF(ts.obs_date, t.disc_date) as `num_days` FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id WHERE t.name LIKE '2022xxf' ORDER BY ts.obs_date DESC LIMIT 1;",

"How many days after SN 2022xxf's discovery was the first spectrum taken?":"SELECT DATEDIFF(ts.obs_date, t.disc_date) as `num_days`  FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id WHERE t.name LIKE '2022xxf' ORDER BY ts.obs_date ASC LIMIT 1;",

"What instrument was used to take each spectra for 2020oi?":"SELECT DISTINCT t.name as `transient_name`, ts.obs_date, i.name as `spectroscopic_instrument` FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id INNER JOIN YSE_App_instrument i ON i.id = ts.instrument_id WHERE t.name LIKE '2020oi';",

"When was the last spectrum taken for SN 2023bee?": "SELECT DISTINCT ts.obs_date FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id WHERE t.name LIKE '2023bee' ORDER BY ts.obs_date DESC LIMIT 1;",

"When was the first spectrum taken for SN 2020oi?": "SELECT DISTINCT ts.obs_date FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id WHERE t.name LIKE '2020oi' ORDER BY ts.obs_date ASC LIMIT 1;",

"Get me the first spectrum taken for SN 2023ixf.": "SELECT DISTINCT tsd.wavelength, tsd.flux, tsd.wavelength_err, tsd.flux_err FROM (SELECT DISTINCT MIN(ts.obs_date) AS `first_spec_date`, t.id FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id INNER JOIN YSE_App_instrument i ON i.id = ts.instrument_id WHERE t.name LIKE '2023ixf' AND ts.obs_date = g.first_spec_date;",

"Get me the last spectrum taken for SN 2023bee.": "SELECT DISTINCT tsd.wavelength, tsd.flux, tsd.wavelength_err, tsd.flux_err FROM (SELECT DISTINCT MAX(ts.obs_date) AS `last_spec_date`, t.id FROM YSE_App_transient t INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_transientspectrum ts ON ts.transient_id = t.id INNER JOIN YSE_App_transientspecdata tsd ON tsd.spectrum_id = ts.id INNER JOIN YSE_App_instrument i ON i.id = ts.instrument_id WHERE t.name LIKE '2023bee' AND ts.obs_date = g.last_spec_date;",

"Who could I reach out to about the KAST spectrograph?": "SELECT DISTINCT pi.name as `pi_name`, pi.phone as `pi_phone`, pi.email as `pi_email`, pi.institution as `pi_institution`, i.name as `instrument_name`, i.description as `instrument_description` FROM YSE_App_principalinvestigator pi INNER JOIN YSE_App_classicalresource cr ON cr.principal_investigator_id = pi.id INNER JOIN YSE_App_instrument i ON  i.telescope_id= cr.telescope_id WHERE i.name = 'KAST';",

"What classical observing programs are ongoing?": "SELECT DISTINCT pi.name,cr.begin_date_valid, cr.end_date_valid, i.name as `instrument_name`, i.description as `instrument_description` FROM YSE_App_principalinvestigator pi INNER JOIN YSE_App_classicalresource cr ON cr.principal_investigator_id = pi.id INNER JOIN YSE_App_instrument i ON  i.telescope_id= cr.telescope_id WHERE TO_DAYS(cr.end_date_valid) > TO_DAYS(CURDATE());",

"Does anyone currently have any follow-up time with DECam?": "SELECT DISTINCT pi.name,cr.begin_date_valid, cr.end_date_valid, i.name as `instrument_name`, i.description as `instrument_description` FROM YSE_App_principalinvestigator pi INNER JOIN YSE_App_classicalresource cr ON cr.principal_investigator_id = pi.id INNER JOIN YSE_App_instrument i ON  i.telescope_id= cr.telescope_id WHERE TO_DAYS(cr.end_date_valid) > TO_DAYS(CURDATE()) AND i.name LIKE 'DECam';",

"Who has existing classical follow-up programs?":"SELECT DISTINCT pi.name FROM YSE_App_principalinvestigator pi INNER JOIN YSE_App_classicalresource cr ON cr.principal_investigator_id = pi.id INNER JOIN YSE_App_instrument i ON  i.telescope_id= cr.telescope_id WHERE TO_DAYS(cr.end_date_valid) > TO_DAYS(CURDATE());",

"What classification is on TNS for SN2023bee?":"SELECT t.name, t.TNS_spec_class FROM YSE_App_transient t WHERE t.name LIKE '2023bee';",

"Give me all transients with UVOT data.":"SELECT DISTINCT t.* FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON t.id = tp.transient_id INNER JOIN YSE_App_transientphotdata tpd ON tp.id = tpd.photometry_id INNER JOIN YSE_App_instrument i ON i.id = tp.instrument_id WHERE i.name = 'UVOT';",

"Get me a list of all transients in YSE Field 253 (A through F).":"""
        SELECT t.*, h.photo_z, ts.name FROM YSE_App_transient t INNER JOIN YSE_App_host h ON h.id = t.host_id WHERE
		((t.ra > (SELECT s.ra_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.A') - 1.65 AND t.ra < (SELECT s.ra_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.A') + 1.65 AND
		t.dec > (SELECT s.dec_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.A') - 1.65 AND t.dec < (SELECT s.dec_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.A') + 1.65) OR
        (t.ra > (SELECT s.ra_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.B') - 1.65 AND t.ra < (SELECT s.ra_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.B') + 1.65 AND
		t.dec > (SELECT s.dec_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.B') - 1.65 AND t.dec < (SELECT s.dec_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.B') + 1.65) OR
        (t.ra > (SELECT s.ra_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.C') - 1.65 AND t.ra < (SELECT s.ra_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.C') + 1.65 AND
		t.dec > (SELECT s.dec_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.C') - 1.65 AND t.dec < (SELECT s.dec_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.C') + 1.65) OR
        (t.ra > (SELECT s.ra_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.D') - 1.65 AND t.ra < (SELECT s.ra_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.D') + 1.65 AND
		t.dec > (SELECT s.dec_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.D') - 1.65 AND t.dec < (SELECT s.dec_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.D') + 1.65) OR
        (t.ra > (SELECT s.ra_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.E') - 1.65 AND t.ra < (SELECT s.ra_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.E') + 1.65 AND
		t.dec > (SELECT s.dec_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.E') - 1.65 AND t.dec < (SELECT s.dec_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.E') + 1.65) OR
        (t.ra > (SELECT s.ra_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.F') - 1.65 AND t.ra < (SELECT s.ra_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.F') + 1.65 AND
		t.dec > (SELECT s.dec_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.F') - 1.65 AND t.dec < (SELECT s.dec_cen FROM YSE_App_surveyfield s
  		WHERE s.field_id = '253.F') + 1.65))
        """,

"When does Brout's DECam program end?":"SELECT DISTINCT pi.name, cr.end_date_valid FROM YSE_App_principalinvestigator pi INNER JOIN YSE_App_classicalresource cr ON cr.principal_investigator_id = pi.id INNER JOIN YSE_App_instrument i ON  i.telescope_id= cr.telescope_id WHERE i.name LIKE 'DECam' AND TO_DAYS(cr.end_date_valid) > TO_DAYS(CURDATE());",

"What is the pan-starrs object ID for the host galaxy of SN2020oi?": "SELECT t.name,  h.name as `host_name`, h.panstarrs_objid FROM YSE_App_transient t INNER JOIN YSE_App_host h on t.host_id = h.id WHERE t.name LIKE '2020oi';",

"What is the photometric redshift (with uncertainty) of SN 2021fqv?": "SELECT t.name, h.photo_z, h.photo_z_err FROM YSE_App_transient t INNER JOIN YSE_App_host h on t.host_id = h.id WHERE t.name LIKE '2021fqv';",

"What are the coordinates (in degrees) of SN 1987A?":"SELECT DISTINCT t.ra, t.`dec` FROM YSE_App_transient t WHERE t.name LIKE '1987A';",

"Give me the names of all spectroscopically-confirmed supernovae discovered after March 21st, 2023.":"SELECT t.name FROM YSE_App_transient t WHERE t.TNS_spec_class IS NOT NULL AND TO_DAYS(t.disc_date) > TO_DAYS('2023-03-21');",

"Select all transients discovered within 3 days of January 10th, 2022.":"SELECT DISTINCT t.name, t.disc_date FROM YSE_App_transient t WHERE ABS(TO_DAYS('2022-01-10') - TO_DAYS(t.disc_date)) <= 3;",

"How many supernovae have a redshift between 0.1 and 0.3? Assume photometric OR spectroscopic redshifts.":" SELECT COUNT(t.name) AS num_supernovae FROM YSE_App_transient t INNER JOIN YSE_App_host h on t.host_id = h.id WHERE (t.redshift > 0.1 AND t.redshift < 0.3) OR (h.photo_z > 0.1 AND h.photo_z < 0.3);",

"What transients peaked brighter than 17th magnitude? Order the results by discovery date.":"SELECT DISTINCT t.name, t.ra AS transient_RA, t.`dec` AS transient_Dec, t.disc_date AS disc_date, pd.mag, t.TNS_spec_class AS spec_class, t.redshift AS transient_z FROM YSE_App_transient t, YSE_App_transientphotdata pd, YSE_App_transientphotometry p WHERE  pd.photometry_id = p.id AND pd.id = (SELECT pd2.id FROM YSE_App_transientphotdata pd2 JOIN YSE_App_transientphotometry p2 ON pd2.photometry_id = p2.id LEFT JOIN YSE_App_transientphotdata_data_quality pdq ON pdq.transientphotdata_id = pd2.id WHERE p2.transient_id = t.id AND pdq.dataquality_id IS NULL AND ISNULL(pd2.mag) = False ORDER BY pd2.mag ASC LIMIT 1)  AND pd.mag < 17 ORDER BY t.disc_date DESC",

"Get me the names and properties of all spectroscopically-confirmed tidal disruption events (TDEs)":"SELECT t.name, t.ra AS transient_RA, t.`dec` AS transient_Dec, t.disc_date AS disc_date, t.redshift AS ‘redshift’,  t.TNS_spec_class FROM YSE_App_transient t WHERE t.TNS_spec_class ='TDE';",

"Give me all spectroscopic classifications.":"SELECT t.TNS_spec_class FROM YSE_App_transient t WHERE t.TNS_spec_class IS NOT NULL;",

"What classification does TNS give for 2023bee?":"SELECT DISTINCT t.name, t.TNS_spec_class FROM YSE_App_transient t WHERE t.name LIKE '2023bee';",

"Get me the name of the closest spectroscopically-confirmed supernova discovered in 2023.":"SELECT DISTINCT t.name, t.TNS_spec_class, t.redshift FROM YSE_App_transient t WHERE t.TNS_spec_class IS NOT NULL AND t.redshift IS NOT NULL AND t.name like '2023%' ORDER BY t.redshift ASC LIMIT 1;",

"What is the closest transient in the database?":"SELECT t.name, t.TNS_spec_class, t.redshift FROM YSE_App_transient t WHERE t.redshift IS NOT NULL ORDER BY t.redshift ASC LIMIT 1;",

"What is the most local transient?":"SELECT t.name, t.TNS_spec_class, t.redshift FROM YSE_App_transient t WHERE t.redshift IS NOT NULL ORDER BY t.redshift ASC LIMIT 1;",

"Which transient is nearest??":"SELECT t.name, t.TNS_spec_class, t.redshift FROM YSE_App_transient t WHERE t.redshift IS NOT NULL ORDER BY t.redshift ASC LIMIT 1;",

"Get me the name of the most distant (or furthest) supernova in the database.":"SELECT DISTINCT t.name, t.redshift FROM YSE_App_transient t WHERE (t.redshift IS NOT NULL) ORDER BY t.redshift DESC LIMIT 1;",

"What does the Transient Name Server classify 2023bee as?":"SELECT DISTINCT t.name, t.TNS_spec_class FROM YSE_App_transient t WHERE t.name LIKE '2023bee';",

"Get the spectroscopic classifications of 10 randomly-selected transients in the database.":"SELECT DISTINCT t.name, t.TNS_spec_class FROM YSE_App_transient t WHERE t.TNS_spec_class IS NOT NULL ORDER BY RAND() LIMIT 10;",

"What is the name and coordinates of the galaxy that hosted supernova 2023bee?":"SELECT h.name, h.ra, h.`dec` FROM YSE_App_host h INNER JOIN YSE_App_transient t ON t.host_id = h.id WHERE t.name LIKE '2023bee';"}

embedder = OpenAIEmbeddings()

few_shot_docs = [
    Document(page_content=question, metadata={"sql_query": few_shots[question]})
    for question in few_shots.keys()
]

len(few_shot_docs)

vector_db = FAISS.from_documents(few_shot_docs, embedder)
vector_db.save_local("/home/gaglian2/theSpeakYSE/data/YSEPZQueries_index_OAI")
