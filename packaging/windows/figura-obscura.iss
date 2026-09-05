; Figura Obscura — Inno Setup installer
;
; Build with:  iscc /DStageDir=..\..\target\stage packaging\windows\figura-obscura.iss
; (packaging\windows\build.ps1 does the whole thing: cargo build, stage, compile.)
;
; The installer's one non-obvious job is the *model download*. Figura Obscura
; cannot censor anything without detector weights, and the weights are far too
; large to embed. Rather than reimplementing a downloader in Pascal Script, the
; final page runs `obscura.exe setup`, which is the same code path the app and the
; CLI use — one implementation, one set of error messages, one thing to test.

#ifndef StageDir
  #define StageDir "..\..\target\stage"
#endif
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#define AppName      "Figura Obscura"
#define AppPublisher "Figura Obscura"
#define AppExeName   "obscura-gui.exe"

[Setup]
AppId={{7C6CFF01-5B3A-4F2E-9D14-5B0C2E7A9E41}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
; Per-user by default: no UAC prompt, and no admin rights needed to buy and run
; a tool from itch.io. `lowest` keeps {autopf} resolving to the user's own
; Programs directory.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\..\target\installer
OutputBaseFilename=FiguraObscura-{#AppVersion}-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\assets\figura-obscura.ico
UninstallDisplayIcon={app}\{#AppExeName}
LicenseFile=..\common\THIRD-PARTY.md
; The app writes only to its own per-user config and cache directories.
UsedUserAreasWarning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "addtopath"; Description: "Add the &command-line tool (obscura.exe) to PATH"; GroupDescription: "Command line:"; Flags: unchecked

[Files]
Source: "{#StageDir}\obscura-gui.exe";  DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\obscura.exe";      DestDir: "{app}"; Flags: ignoreversion
; Bundled ffmpeg — ob-media looks in {app}\bin before falling back to PATH.
Source: "{#StageDir}\bin\*";       DestDir: "{app}\bin"; Flags: ignoreversion skipifsourcedoesntexist
; GPU execution providers, present only in a CUDA build.
Source: "{#StageDir}\onnxruntime_providers_*"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StageDir}\THIRD-PARTY.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\README.md";      DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\licenses\*";     DestDir: "{app}\licenses"; Flags: ignoreversion skipifsourcedoesntexist recursesubdirs

[Icons]
Name: "{group}\{#AppName}";            Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}";  Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";      Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; Only touched when the user opts in; removed cleanly on uninstall.
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
    ValueData: "{olddata};{app}"; Check: NeedsAddPath(ExpandConstant('{app}')); Tasks: addtopath

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The model cache is deliberately NOT removed: it is user data, it is expensive
; to re-download, and a reinstall should find it still there.
Type: filesandordirs; Name: "{app}\bin"

[Code]
var
  ModelPage: TInputOptionWizardPage;
  DownloadFailed: Boolean;

{ True if Dir is not already a component of the user's PATH. }
function NeedsAddPath(Dir: string): Boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(OrigPath) + ';') = 0;
end;

procedure InitializeWizard;
begin
  ModelPage := CreateInputOptionPage(wpSelectTasks,
    'Detection models',
    'Figura Obscura needs a detection model before it can censor anything.',
    'Models are about 56 MB in total. They are downloaded once and then used ' +
    'entirely offline — nothing you process ever leaves your computer.' + #13#10#13#10 +
    'If you skip this, you can download them any time from the Models page ' +
    'inside the app.',
    False, False);
  ModelPage.Add('Download the recommended models now (requires an internet connection)');
  ModelPage.Values[0] := True;
end;

{ Run `obscura setup` after the files are in place.

  ssPostInstall rather than a [Run] entry so a failed download can be reported
  on the wizard's own progress page and the install can still complete: a
  machine that is offline during setup must still end up with a working
  installation, just without models yet. }
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then
    exit;
  if not ModelPage.Values[0] then
    exit;

  WizardForm.StatusLabel.Caption := 'Downloading detection models…';
  WizardForm.FilenameLabel.Caption := 'This can take a few minutes.';

  { --quiet: no progress bars, because the output goes to a hidden console. }
  if not Exec(ExpandConstant('{app}\obscura.exe'), 'setup --quiet', '',
              SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    DownloadFailed := True
  else
    DownloadFailed := ResultCode <> 0;

  if DownloadFailed then
    MsgBox('The detection models could not be downloaded.' + #13#10#13#10 +
           'Figura Obscura is installed and will still start — open the Models ' +
           'page inside the app to try again once you are online.',
           mbInformation, MB_OK);
end;
