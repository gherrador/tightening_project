from typing import List

COLUMNS_TO_DROP: List[str] = [
    "FCT_ID", "FCT_Label", "FCT_Version",
    "LIN_Version",
    "WKA_ID", "WKA_Label",
    "CTRL_Name", "CTRL_IPAddress", "CTRL_Type", "CTRL_Version",
    "TU_Number", "TU_Comment",
    "PS_ID", "PS_Number", "PS_TorqueUnit", "PS_Version",
    "STEP_Number", "STEP_Type", "STEP_AbortTorque",
    "STEP_AngleThreshold", "STEP_AngleTarget", "STEP_AngleMinTolerance", "STEP_AngleMaxTolerance",
    "STEP_AbortAngle",
    "STEP_TorqueRateTarget", "STEP_TorqueRateMinTolerance", "STEP_TorqueRateMaxTolerance",
    "TOOL_SerialNumber",
    "RES_ID", "RES_ResultNumber", "RES_StepNumber", "RES_RecordType",
    "RES_ResultTypeId", "RES_ControllerSerialNumber", "RES_CableSerialNumber",
    "RES_Report", "RES_VIN", "RES_CycleOKCount", "RES_Time", "RES_ErrorCode",
    "RES_StopSource", "RES_TorqueAccuracy", "RES_TorqueTrend",
    "RES_FinalAngle", "RES_AngleTrend",
    "RES_FinalTorqueRate", "RES_TorqueRateTrend",
    "RES_MinCurrent", "RES_MaxCurrent", "RES_FinalCurrent",
    "RES_FinalCurrent_Percent", "RES_MinCurrent_Percent", "RES_MaxCurrent_Percent",
    "RES_CurrentTrend",
    "RES_SecondTransducerMode", "RES_SecondFinalTorque", "RES_SecondFinalAngle",
    "RES_SecondTransducer_TorqueDev", "RES_SecondTransducer_AngleDev",
    "WKA_Version",
    "CTRL_ID",
]


FINAL_COLUMNS_ORDER: List[str] = [
    "LIN_ID", "LIN_Label",
    "TU_ID", "TU_Name",
    "PS_Comment",
    "Tool_ID", "TOOL_Number", "TOOL_Comment",
    "DateTime",
    "STEP_ID",
    "STEP_TorqueTarget", "STEP_TorqueMinTolerance", "STEP_TorqueMaxTolerance",
    "FinalTorque",
    "Torque_Result",
    "Quality_Status",
]


RENAME_MAP = {
    "STEP_TorqueTarget": "TorqueTarget", 
    "STEP_TorqueMinTolerance": "TorqueMinTolerance", 
    "STEP_TorqueMaxTolerance": "TorqueMaxTolerance",
    "RES_DateTime": "DateTime",
    "RES_FinalTorque": "FinalTorque",
}
