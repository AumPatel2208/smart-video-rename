; Inno Setup script — produces a Windows .exe installer.
; Run after scripts\build.bat: iscc scripts\installer.iss

[Setup]
AppName=Smart Video Rename
AppVersion=1.0.0
AppPublisher=Smart Video Rename
DefaultDirName={autopf}\SmartVideoRename
DefaultGroupName=Smart Video Rename
OutputDir=dist
OutputBaseFilename=SmartVideoRename-Windows-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "dist\SmartVideoRename\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\Smart Video Rename"; Filename: "{app}\SmartVideoRename.exe"
Name: "{commondesktop}\Smart Video Rename"; Filename: "{app}\SmartVideoRename.exe"

[Run]
Filename: "{app}\SmartVideoRename.exe"; Description: "Launch Smart Video Rename"; Flags: nowait postinstall skipifsilent
