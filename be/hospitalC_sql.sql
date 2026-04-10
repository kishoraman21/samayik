-- Hospital C — Patient vitals exported from legacy SQL database
-- This is what many hospitals export when they dump their records

INSERT INTO patient_records (patient_id, bp_systolic, bp_diastolic, pulse, body_temperature, resp_rate, spo2_level, weight, allergy_info, prescribed_drugs, mental_health_score)
VALUES ('P004', 142, 92, 95, 38.2, 22, 94, 85, 'Sulfa drugs', 'Amlodipine, Metoprolol', 14);

INSERT INTO patient_records (patient_id, bp_systolic, bp_diastolic, pulse, body_temperature, resp_rate, spo2_level, weight, allergy_info, prescribed_drugs, mental_health_score)
VALUES ('P005', 118, 76, 68, 36.9, 16, 98, 62, NULL, 'Levothyroxine', NULL);

INSERT INTO patient_records (patient_id, bp_systolic, bp_diastolic, pulse, body_temperature, resp_rate, spo2_level, weight, allergy_info, prescribed_drugs, mental_health_score)
VALUES ('P006', 130, 85, 78, 37.0, 18, 97, 70, 'Penicillin, Latex', 'Lisinopril, Atorvastatin, Aspirin', 8);
