; Inno Setup Skript fuer Audio Visualizer Pro.
;
; Baut einen Per-User-Installer (kein Admin noetig) aus der onedir-Ausgabe
; von PyInstaller (dist/AudioVisualizerPro/). Erst build/build.py ausfuehren,
; dann dieses Skript kompilieren:
;
;   ISCC build/installer.iss /DMyAppVersion=3.0.0
;
; MyAppVersion per Kommandozeile setzen (aus pyproject.toml), damit die
; Version nicht doppelt gepflegt werden muss. Ohne /D-Flag greift der
; Fallback-Wert unten (bei Versions-Bumps mit aktualisieren).

#ifndef MyAppVersion
  #define MyAppVersion "3.0.0"
#endif

#define MyAppName "Audio Visualizer Pro"
#define MyAppPublisher "Audio Visualizer Pro Team"
#define MyAppExeName "AudioVisualizerPro.exe"
#define MyDistDir "..\dist\AudioVisualizerPro"

[Setup]
AppId={{B4E6C8A0-9F3D-4C7B-9E2A-3D8F1A6B5C90}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Per-User-Installation: kein Admin-Dialog, kein UAC-Prompt.
PrivilegesRequired=lowest
DefaultDirName={userpf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist\installer
OutputBaseFilename=AudioVisualizerPro-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; Kein SetupIconFile bis assets/icon.ico existiert (siehe docs/INSTALLATION.md).
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "german"; MessagesFile: "compiler:Languages\German.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; onedir-Ausgabe komplett mitnehmen (exe + _internal/ mit allen Abhaengigkeiten)
Source: "{#MyDistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[Code]
// Nutzerdaten (Rezepte, Cache, Logs, FFmpeg-Download) liegen bewusst ausserhalb
// des Install-Ordners (%APPDATA%/%LOCALAPPDATA%\AudioVisualizerPro), siehe
// src/paths.py. Der Uninstaller loescht sie NICHT automatisch — stattdessen
// wird nachgefragt, damit gespeicherte Studio-Rezepte nicht versehentlich
// verloren gehen.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDataDir, LocalAppDataDir: String;
  Response: Integer;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    AppDataDir := ExpandConstant('{userappdata}\AudioVisualizerPro');
    LocalAppDataDir := ExpandConstant('{localappdata}\AudioVisualizerPro');

    if DirExists(AppDataDir) or DirExists(LocalAppDataDir) then
    begin
      Response := MsgBox(
        'Auch gespeicherte Nutzerdaten loeschen (Studio-Rezepte, Cache, Logs, ' +
        'heruntergeladenes FFmpeg)?' #13#13 +
        AppDataDir #13 + LocalAppDataDir,
        mbConfirmation, MB_YESNO
      );
      if Response = IDYES then
      begin
        DelTree(AppDataDir, True, True, True);
        DelTree(LocalAppDataDir, True, True, True);
      end;
    end;
  end;
end;
