Dim fullPath, scriptDir, py, script, shell
fullPath = WScript.ScriptFullName
scriptDir = Left(fullPath, InStrRev(fullPath, "\"))
py = scriptDir & "venv\Scripts\pythonw.exe"
script = scriptDir & "yolo_cam.py"
Set shell = CreateObject("WScript.Shell")
shell.Environment("PROCESS")("YOLO_START_HIDDEN") = "0"
shell.Environment("PROCESS")("YOLO_DISABLE_TUNNEL") = "0"
shell.Run """ & py & "" "" & script & """, 0, False
