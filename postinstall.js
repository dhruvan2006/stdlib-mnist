const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

(async () => {
    console.log('Starting @stdlib native patch...');

    const rootDir = process.cwd();
    const rootPkg = JSON.parse(fs.readFileSync(path.join(rootDir, 'package.json'), 'utf8'));
    const stdlibDeps = Object.keys(rootPkg.dependencies || {}).filter(dep => dep.startsWith('@stdlib/'));

    console.log('Scanning for all native dependencies across all packages...');
    const allRequiredDeps = new Set();
    
    for (const pkgName of stdlibDeps) {
        const shortName = pkgName.replace('@stdlib/', '');
        const pkgVersion = rootPkg.dependencies[pkgName].replace('^', '').replace('~', '');
        
        try {
            const baseUrl = `https://raw.githubusercontent.com/stdlib-js/${shortName}/v${pkgVersion}`;
            const pkgJsonRes = await fetch(`${baseUrl}/package.json`);
            if (pkgJsonRes.ok) {
                const remotePkg = await pkgJsonRes.json();
                Object.keys(remotePkg.devDependencies || {}).forEach(d => {
                    if (d.startsWith('@stdlib/')) allRequiredDeps.add(d);
                });
            }
        } catch (e) { /* skip */ }
    }

    const missing = Array.from(allRequiredDeps).filter(d => !fs.existsSync(path.join(rootDir, 'node_modules', d)));
    if (missing.length > 0) {
        console.log(`Installing ${missing.length} missing support modules...`);
        execSync(`npm install ${missing.join(' ')} --no-save`, { stdio: 'inherit' });
    }

    for (const pkgName of stdlibDeps) {
        const shortName = pkgName.replace('@stdlib/', '');
        const pkgVersion = rootPkg.dependencies[pkgName].replace('^', '').replace('~', '');
        const pkgDir = path.join(rootDir, 'node_modules', pkgName);

        if (!fs.existsSync(pkgDir)) continue;

        try {
            const baseUrl = `https://raw.githubusercontent.com/stdlib-js/${shortName}/v${pkgVersion}`;
            const bindingRes = await fetch(`${baseUrl}/binding.gyp`);
            if (bindingRes.status === 404) continue;

            console.log(`Patching: ${pkgName}...`);
            const bindingGypContent = await bindingRes.text();
            const includeRes = await fetch(`${baseUrl}/include.gypi`);
            let includeGypiContent = await includeRes.text();

            const absManifestPath = path.join(pkgDir, 'manifest.json').replace(/\\/g, '/');
            const absBaseDir = pkgDir.replace(/\\/g, '/');

            includeGypiContent = includeGypiContent
                .replace(/\\'\\.\/manifest\.json\\'/g, `\\'${absManifestPath}\\'`)
                .replace(/process\.cwd\(\)/g, `\\'${absBaseDir}\\'`);

            fs.writeFileSync(path.join(pkgDir, 'binding.gyp'), bindingGypContent);
            fs.writeFileSync(path.join(pkgDir, 'include.gypi'), includeGypiContent);

            console.log(`Rebuilding ${pkgName}...`);
            execSync('npx node-gyp rebuild', { stdio: 'inherit', cwd: pkgDir });
            
        } catch (err) {
            console.error(`Failed ${pkgName}: ${err.message}`);
        }
    }
    console.log('\nDone! Try running "npm run train" now.');
})();