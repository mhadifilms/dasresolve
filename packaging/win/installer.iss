; Copyright 2026 The DasGrain OFX Authors.
; SPDX-License-Identifier: Apache-2.0
;
; Inno Setup script for the DasGrain OFX plugin. Build the plugin first
; (cmake --build build --config Release) so build/DasGrain.ofx.bundle
; exists, then run this script through the Inno Setup compiler:
;
;     "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" \
;          packaging\win\installer.iss

#define MyAppName "DasGrain"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "DasGrain OFX Authors"
#define MyAppURL "https://github.com/"
#define MyAppExeName "DasGrain.ofx"
#define BundleSrcDir "..\..\build\DasGrain.ofx.bundle"

[Setup]
AppId={{C0F6CDC0-DA3D-4FCD-B7F0-DAE93EE0C1B7}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={commoncf}\OFX\Plugins\DasGrain.ofx.bundle
DefaultGroupName={#MyAppName}
DisableDirPage=yes
OutputBaseFilename=DasGrain-Installer
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "{#BundleSrcDir}\Contents\Info.plist";              DestDir: "{app}\Contents"
Source: "{#BundleSrcDir}\Contents\Resources\LICENSE.txt";   DestDir: "{app}\Contents\Resources"; Flags: ignoreversion
Source: "{#BundleSrcDir}\Contents\Resources\NOTICE.txt";    DestDir: "{app}\Contents\Resources"; Flags: ignoreversion
Source: "{#BundleSrcDir}\Contents\Resources\OPENFX-LICENSE.txt"; DestDir: "{app}\Contents\Resources"; Flags: ignoreversion
Source: "{#BundleSrcDir}\Contents\Win64\{#MyAppExeName}";   DestDir: "{app}\Contents\Win64"; Flags: ignoreversion

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  if not DirExists(ExpandConstant('{commoncf}\OFX\Plugins')) then
    ForceDirectories(ExpandConstant('{commoncf}\OFX\Plugins'));
end;
