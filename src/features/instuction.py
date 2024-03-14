""" Data Structure """
# TODO
#  dataPreprocessed =
#   {
#       ID_1: {
#           HOSP_ADMIT_TIME: 0,                         <INIT_TIME: Relative time, which is ALWAYS 0>
#           HOSP_DISCHRG_HRS_FROM_ADMIT: time,          <END_TIME: Relative time>
#           WEIGHT_KG: float,
#           HEIGHT_CM: float,
#           AGE: int,
#           SEX: int,
#           DISEASES: [                                 * ENCODING: ONE-HOT APPLIED, TOTAL_VARIETIES=150 *
#               diseaseID_1: int (idx),
#               diseaseID_2: int (idx),
#               ...
#           ],
#           OR_PROC_ID: [                               * ENCODING: ONE-HOT APPLIED, TOTAL_VARIETIES=108 *
#               OR_PROC_ID_01,
#               OR_PROC_ID_02,
#               ...
#           ],
#           ORDERS_ACTIVITY_START_TIME: [               <ACTIVITY_TIME: Relative start time, all activities have same ID -> ignored>
#               activity_01_startTime: time,
#               activity_02_startTime: time,
#               ...
#           ],
#           ORDERS_ACTIVITY_STOP_TIME: [                <ACTIVITY_TIME: Relative end time>
#               activity_01_stopTime: time,
#               activity_02_stopTime: time,
#               ...
#           ],
#           ORDERS_NUTRITION: [                         * ENCODING: ONE-HOT APPLIED, TOTAL_VARIETIES=15 *
#               nutrition_01: int,
#               nutrition_02: int,
#               ...
#           ],
#           ORDERS_NUTRITION_START_TIME: [              <NUTRITION_TIME: Relative start time>
#               nutrition_01_startTime: time,
#               nutrition_02_startTime: time,
#               ...
#           ],
#           ORDERS_NUTRITION_STOP_TIME: [               <NUTRITION_TIME: Relative end time>
#               nutrition_01_stopTime: time,
#               nutrition_02_stopTime: time,
#               ...
#           ],
#           LAB_RESULT_HRS_FROM_ADMIT: [                <TEST_TIME: Relative time>
#               LAB_RESULT_TIME_1: time,
#               LAB_RESULT_TIME_2: time,
#               ...
#           ],
#           LAB_COMPONENT_ID: [                         * ENCODING: ONE-HOT APPLIED, TOTAL_VARIETIES=16 *
#               LAB_COMPONENT_TIME_1: time,
#               LAB_COMPONENT_TIME_2: time,
#               ...
#           ],
#           LAB_ORD_VALUE: [                            ?????
#               LAB_ORD_VALUE_TIME_1: float，
#               LAB_ORD_VALUE_TIME_2: float，
#               ...
#           ],
#           MEDICATION_ATC: [
#               MEDICATION_ATC_1: str;
#               MEDICATION_ATC_2: str;
#               ...
#           ],
#           MEDICATION_ATC_ENCODED: [                   * (NEW) ENCODING: Doc2Vec APPLIED -> `DRUGS_d2vModel.pkl` *
#               <Doc2Vec_ENCODED_VECTOR, Size=5>
#               <Same as MEDICATION_ATC (kept for time checking)>
#           ],
#           MEDICATION_TAKEN_HRS_FROM_ADMIT: [          <TREATMENT_TIME: Relative time>
#               MEDICATION_TAKEN_TIME_1: time;
#               MEDICATION_TAKEN_TIME_2: time;
#               ...
#           ],
#           MEDICATION_SIG: [                           ?????
#               MEDICATION_SIG_1: float;
#               MEDICATION_SIG_2: float;
#               ...
#           ],
#           MEDICATION_ACTIONS: [                       * ENCODING: Doc2Vec *
#               MEDICATION_ACTION_1: str;
#               MEDICATION_ACTION_1: str;
#               ...
#           ],
#           MEDICATION_ACTIONS_ENCODED: [               * ENCODING: Doc2Vec APPLIED -> `ACTIONS_d2vModel.pkl` *
#               <Doc2Vec_ENCODED_VECTOR, Size=5>
#               <Same as MEDICATION_ACTIONS (kept for time checking)>
#           ],
#           PRIOR_MEDICATION_ATC_ENCODED: [             * ENCODING: Doc2Vec APPLIED -> `DRUGS_d2vModel.pkl` *
#              <Doc2Vec_ENCODED_VECTOR, Size=5>
#           ],
#           PRIOR_MEDICATION_DISP_DAYS_NORM: [          * ENCODING: ONE-HOT, TOTAL_VARIETIES=10, just as features not time *
#               PRIOR_MEDICATION_DISP_DAYS_NORM_1: int;     --> [0, 10], indicates how far the time prior
#               PRIOR_MEDICATION_DISP_DAYS_NORM_2: int;
#               ...
#           ],
#       },
#       ID_2: {
#           ...
#       },
#       ...
#   }

""" Sequential Logics """
# TODO
#  ------> ADMIT_TIME ------> NUTRITION/ACTIVITY/TREATMENT/TEST_TIME ------> DISCHRG_TIME ------>
#     [INIT, 0_hr]       [Comparing to ADMIT_TIME, i_hr, 0 < i < n]      [END, n_hr, Comparing to ADMIT_TIME]

""" Dataset Explanations """
# TODO
#  1. DATA: Contains all personal features and event time records for each sample case;
#  2. FEATURE_DATA: Contains only personal features for each sample case, ordered by `category`;
#  3. SEQUENCE_DATA: Contains only event time records for each sample case, ordered by `time`;
