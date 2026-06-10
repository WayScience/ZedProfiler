import { useEffect, useRef } from "react";
import { useEventListener } from "@reactuses/core";
import { proxy, useSnapshot } from "valtio";
import gsap from "gsap";
import * as three from "three";
import * as fiber from "@react-three/fiber";
import * as drei from "@react-three/drei";
import * as postprocessing from "@react-three/postprocessing";
import { BlendFunction } from "postprocessing";
import { Recorder, RecorderStatus } from "canvas-record";

const blue = "hsl(200, 100%, 60%)";
const pink = "hsl(340, 100%, 70%)";
const purple = "hsl(240, 35%, 25%)";

const fps = 60;

const tau = 2 * Math.PI;
let recorder: Recorder;
const step = 1 / fps;
let time = 0;

gsap.ticker.fps(fps);
gsap.ticker.remove(gsap.updateRoot);

const animation = proxy({
  spin: 0,
  wave: Array(6)
    .fill(0)
    .map((value) => ({ value })),
});
const timeline = gsap.timeline();
timeline.to(animation, { spin: tau, ease: "linear", duration: 6 });
animation.wave.forEach((wave, index) =>
  timeline
    .to(
      wave,
      { value: 0.1, ease: "power1.inOut", duration: 0.75 },
      1 + index / 3,
    )
    .to(wave, { value: 0, ease: "elastic.out", duration: 5 }, ">"),
);

const start = async (ctx: WebGLRenderingContext) => {
  if (recorder) return;
  recorder = new Recorder(ctx, {
    extension: "mp4",
    duration: 6,
    frameRate: fps,
    encoderOptions: { bitrate: 100_000_000 },
  });
  await recorder.start();
  gsap.updateRoot((time += step));
};

const frame = async () => {
  if (!recorder) return;
  if (recorder.status === RecorderStatus.Recording) {
    if (timeline.time() < timeline.duration()) {
      await recorder.step();
      gsap.updateRoot((time += step));
    } else recorder.stop();
  }
};

const size = 2000 / window.devicePixelRatio;

export default function App() {
  return (
    <fiber.Canvas
      style={{ width: size, height: size, border: "1px solid black" }}
      camera={{ position: [0, 0, 7], fov: 45 }}
      gl={{
        toneMapping: three.ACESFilmicToneMapping,
        antialias: true,
        preserveDrawingBuffer: true,
      }}
    >
      <Canvas />
    </fiber.Canvas>
  );
}

function Canvas() {
  const { spin, wave } = useSnapshot(animation);

  const { gl } = fiber.useThree();

  useEventListener("keydown", ({ key }) => {
    if (key === "r") start(gl.getContext());
  });

  useEffect(() => {
    const timer = window.setTimeout(frame, 1000 * step);
    return () => window.clearTimeout(timer);
  });

  return (
    <>
      <drei.Environment>
        <group rotation={[0, -spin, 0]}>
          <drei.Lightformer
            position={[-2, 2, 0]}
            scale={5}
            intensity={10}
            color="white"
          />
          <drei.Lightformer
            position={[2, -2, 0]}
            scale={5}
            intensity={5}
            color="white"
          />
        </group>
      </drei.Environment>

      <Sphere
        color={purple}
        position={[0, 0, 0]}
        scale={2.5}
        bumpy={0}
        order={-1}
      />

      <DepthClear renderOrder={0} />

      <group rotation={[0, 0, tau / 6]}>
        <group rotation={[0, 0, 0]}>
          <Sphere position={polar(2 + wave[0].value, 65)} scale={0.05} />
          <Sphere position={polar(2 + wave[1].value, 52.5)} scale={0.1} />
          <Sphere position={polar(2 + wave[2].value, 37.5)} scale={0.15} />
          <Sphere position={polar(2 + wave[3].value, 20)} scale={0.2} />
          <Sphere position={polar(2 + wave[4].value, 0)} scale={0.25} />
          <Sphere position={polar(1.25 + wave[5].value, 0)} scale={0.15} />
        </group>

        <group rotation={[0, 0, tau / 2]}>
          <Sphere position={polar(2 + wave[0].value, 65)} scale={0.05} />
          <Sphere position={polar(2 + wave[1].value, 52.5)} scale={0.1} />
          <Sphere position={polar(2 + wave[2].value, 37.5)} scale={0.15} />
          <Sphere position={polar(2 + wave[3].value, 20)} scale={0.2} />
          <Sphere position={polar(2 + wave[4].value, 0)} scale={0.25} />
          <Sphere position={polar(1.25 + wave[5].value, 0)} scale={0.15} />
        </group>
      </group>

      <group rotation={[0, spin, 0]}>
        <Sphere color={pink} position={[0, 0, 0]} scale={0.75} bumpy={0.05} />
      </group>

      {/* <drei.OrbitControls /> */}

      <postprocessing.EffectComposer>
        <postprocessing.Noise
          opacity={0.35}
          blendFunction={BlendFunction.OVERLAY}
        />
      </postprocessing.EffectComposer>
    </>
  );
}

const DepthClear = ({ renderOrder = 0 }) => (
  <mesh
    renderOrder={renderOrder}
    onBeforeRender={(renderer: three.WebGLRenderer) => renderer.clearDepth()}
  >
    <planeGeometry />
    <meshBasicMaterial colorWrite={false} depthWrite={false} />
  </mesh>
);

const Sphere = ({
  color = blue,
  position = [0, 0, 0] as [number, number, number],
  rotation = [0, 0, 0] as [number, number, number],
  scale = 1,
  bumpy = 0,
  order = 1,
}) => {
  const [x, y, z] = position;
  const bumpyRef = useRef({ value: bumpy });
  bumpyRef.current.value = bumpy;

  return (
    <drei.Octahedron
      position={[x, y, z]}
      args={[1, 200]}
      scale={scale}
      rotation={rotation}
      renderOrder={order}
    >
      <meshStandardMaterial
        color={color}
        metalness={1}
        roughness={1}
        onBeforeCompile={(shader: any) => {
          shader.uniforms.s = bumpyRef.current;
          shader.vertexShader = `
uniform float s;
${shader.vertexShader}
`;
          shader.vertexShader = shader.vertexShader.replace(
            "#include <begin_vertex>",
            `#include <begin_vertex>
vec3 v = normalize(position);
float f = 4.0f;
float p = 1.0f;
float x = pow(cos(v.x * f * 3.14159), p);
float y = pow(cos(v.y * f * 3.14159), p);
float z = pow(sin(v.z * f * 3.14159), p);
float scale = x + y + z;
transformed *= 1.0f + scale * s;
`,
          );
        }}
      />
    </drei.Octahedron>
  );
};

const polar = (r = 0, a = 0, z = 0): [number, number, number] => [
  r * Math.cos((a / 180) * Math.PI),
  r * Math.sin((a / 180) * Math.PI),
  z,
];
