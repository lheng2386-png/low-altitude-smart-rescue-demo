import { useState, useEffect, useMemo, useRef } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import {
    ScanEye,
    ScanSearch,
    Car,
    Flame,
    PersonStanding,
    Funnel,
    Info,
    Eye,
    EyeOff,
    Scissors
} from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { getDetectionColor, getDetectionLabel } from '@/types/detection';
import { useQueryClient } from '@tanstack/react-query';
import {
    useStartDetection,
    useDetectionStatusPolling,
    useFetchNewDetections,
    useDetections,
    useIsDetectionRunning,
} from "@/hooks/detectionHooks";
import {
    countDetections,
    initiateThresholds,
    initiateCategoryVisibility,
    updateThresholds,
    updateCategoryVisibility,
} from "@/utils/detectionUtils";
import { useMaps } from "@/hooks/useMaps";


interface Props {
    report_id: number;
    setThresholds: (thresholds: { [key: string]: number }) => void;
    thresholds: { [key: string]: number };
    setFilter: (filter: string[]) => void;
    filters: string[];
    visibleCategories: { [key: string]: boolean };
    setVisibleCategories: (visibility: { [key: string]: boolean }) => void;
    clipDetections: boolean;
    setClipDetections: (v: boolean) => void;
}

export function DetectionCard({ report_id, setThresholds, thresholds, setFilter, filters, visibleCategories, setVisibleCategories, clipDetections, setClipDetections }: Props) {
    const [pollingEnabled, setPollingEnabled] = useState(false);
    const isRunning = useIsDetectionRunning(report_id);
    const { data: detections } = useDetections(report_id);
    const { data: maps } = useMaps(report_id);
    const hasVoronoi = useMemo(() =>
        maps?.some(m => m.map_elements?.some(el => el.voronoi_gps?.length)) ?? false
    , [maps]);
    const queryClient = useQueryClient();
    const detectionSummary = useMemo(() => { if (detections) return countDetections(detections, thresholds); }, [detections, thresholds]);
    const [hasDetections, setHasDetections] = useState(detections && detections.length > 0);
    const [detectionMode, setDetectionMode] = useState<"fast" | "medium" | "detailed" | "experimental" | undefined>(undefined);

    useEffect(() => {
        if (detections) {
            setHasDetections(detections.length > 0);
            if (Object.keys(thresholds).length == 0) {
                setThresholds(initiateThresholds(detections));
                setVisibleCategories(initiateCategoryVisibility(detections));
            }
            else {
                setThresholds(updateThresholds(detections, thresholds));
                setVisibleCategories(updateCategoryVisibility(detections, visibleCategories));
            }
        }
    }, [detections]);

    // enable polling automatically if backend says process is running
    useEffect(() => {
        if (isRunning.data) {
            setPollingEnabled(true);
        }
    }, [isRunning.data]);

    const startDetection = useStartDetection();
    const detectionStatus = useDetectionStatusPolling(report_id, pollingEnabled);
    const fetchNewDetections = useFetchNewDetections(report_id);
    const lastProgressRef = useRef(0);

    useEffect(() => {
        if (!detectionStatus.data) return;

        const progress = detectionStatus.data.progress ?? 0;

        // If progress increased, request more detections
        if (progress > lastProgressRef.current) {
            fetchNewDetections.mutate();
        }
        lastProgressRef.current = progress;

        const normalizedStatus = detectionStatus.data.status.toUpperCase();
        if (normalizedStatus === "FINISHED" || normalizedStatus === "ERROR" || normalizedStatus === "FAILED") {
            setPollingEnabled(false);
            if (normalizedStatus === "FINISHED") {
                // refresh detections by invalidating queries  ["report", report.report_id]
                queryClient.invalidateQueries({ queryKey: ["detections", report_id] });
                console.log("Detections should be updated");
            }
        }
    }, [detectionStatus.data]);


    const handleStart = () => {
        if (!detectionMode) return;

        startDetection.mutate(
            { reportId: report_id, processingMode: detectionMode },
            {
                onSuccess: () => {
                    queryClient.invalidateQueries({ queryKey: ["detections", report_id] });
                    setPollingEnabled(true);
                },
            }
        );
    };


    const selectObjectIcon = (objectType: string) => {
        const color = getDetectionColor(objectType);

        let icon;

        switch (objectType) {
            case 'vehicle':
                icon = <Car className="w-4 h-4" color={"black"} />;
                break;
            case 'fire':
                icon = <Flame className="w-4 h-4" color={"black"} />;
                break;
            case 'human':
                icon = <PersonStanding className="w-4 h-4" color={"black"} />;
                break;
            default:
                icon = <ScanSearch className="w-4 h-4" color={"black"} />;
                break;
        }

        return (
            <div
                className={`w-6 h-6 flex items-center justify-center rounded-sm bg-[var(--tag-color)]  mr-1`}
                style={{ '--tag-color': color }}
            >
                {icon}
            </div>
        );
    };

    const infotextForMode = (mode: string) => {
        switch (mode) {
            case 'fast':
                return "快速模式使用经典救援检测模型直接扫描整张图，速度较快，但依赖模型文件可用。";
            case 'medium':
                return "标准模式使用经典救援检测模型并切分图像，速度和精度较均衡。";
            case 'detailed':
                return "精细模式使用经典救援检测模型进行更多切分，速度较慢，适合高空或小目标。";
            case 'experimental':
                return "YOLOv11 模式使用本地 YOLO 模型，是另一条独立检测管线。";
            default:
                return "暂无该模式说明。";
        }
    };


    return (
        <>
            <Card className="min-w-72 max-w-320 flex-2 relative overflow-hidden pb-3">
                <ScanEye className="absolute right-2 top-1 w-24 h-24 opacity-100 text-muted-foreground dark:text-white z-0 pointer-events-none" />

                {/* Gradient Overlay */}
                <div className="absolute w-40 h-30 right-0 top-0 z-10 pointer-events-none bg-gradient-to-l from-white/90 via-white/75 to-white/55 dark:from-gray-900/100 dark:via-gray-900/85 dark:to-gray-900/60" />

                <CardContent className="px-4 pt-1 flex flex-col items-start space-y-1 relative z-10">
                    {/* Title */}

                    <div className="flex justify-between items-start w-full">
                        <div className="text-xl font-bold leading-none">目标检测</div>
                    </div>

                    {/* Description */}
                    {hasDetections ? (
                        <div className="flex items-center justify-start w-full gap-1 mt-0">
                            <Table className="w-full table-fixed">
                                <TableHeader>
                                    <TableRow>
                                        <TableHead className="w-[52%]">类型</TableHead>
                                        <TableHead className="w-14 text-right">数量</TableHead>
                                        <TableHead className="w-20">阈值</TableHead>
                                        <TableHead className="w-16"></TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {detectionSummary && Object.entries(detectionSummary).map(([key, count]) => (
                                        <TableRow key={key} className="hover:bg-muted transition-colors p-1">
                                            <TableCell className="py-1">
                                                <div className="flex min-w-0 items-center">
                                                    {selectObjectIcon(key)} {getDetectionLabel(key)}
                                                </div>
                                            </TableCell>
                                            <TableCell className="py-1 text-right tabular-nums">
                                                {count}
                                            </TableCell>
                                            <TableCell className="py-1">
                                                <Input
                                                    type="number"
                                                    min="0"
                                                    max="1"
                                                    step="0.01"
                                                    value={thresholds[key]}
                                                    onChange={(e) => {
                                                        const newThresholds = { ...thresholds, [key]: parseFloat(e.target.value) };
                                                        setThresholds(newThresholds);
                                                    }}
                                                    className="h-8 w-full m-0"
                                                />
                                            </TableCell>
                                            <TableCell className="py-1">
                                                <Button
                                                    variant={visibleCategories[key] ? "ghost" : "outline"}
                                                    size="icon"
                                                    className={`p-0 m-0 ml-1 ${visibleCategories[key] ? "outline-white" : ""}`}
                                                    onClick={() => {
                                                        const newVisibility = { ...visibleCategories, [key]: !visibleCategories[key] };
                                                        setVisibleCategories(newVisibility);
                                                    }}
                                                >
                                                    {visibleCategories[key] ? <Eye className="w-4 h-4 p-0 m-0" /> :
                                                        <EyeOff className="w-4 h-4 p-0 m-0" />
                                                    }
                                                </Button>

                                                <Button
                                                    variant={filters.includes(key) ? "default" : "outline"}
                                                    size="icon"
                                                    className='p-0 m-0'
                                                    onClick={() => {
                                                        const isCurrentlyFiltered = filters.includes(key);

                                                        if (isCurrentlyFiltered) {
                                                            // Clear filter and restore all categories to visible
                                                            setFilter([]);
                                                            const newVisibility: { [k: string]: boolean } = {};
                                                            Object.keys(visibleCategories).forEach(cat => {
                                                                newVisibility[cat] = true;
                                                            });
                                                            setVisibleCategories(newVisibility);
                                                        } else {
                                                            // Set filter to this category and hide all others
                                                            setFilter([key]);
                                                            const newVisibility: { [k: string]: boolean } = {};
                                                            Object.keys(visibleCategories).forEach(cat => {
                                                                newVisibility[cat] = cat === key;
                                                            });
                                                            setVisibleCategories(newVisibility);
                                                        }
                                                    }}
                                                >
                                                    <Funnel className="w-4 h-4 p-0 m-0" />
                                                </Button>

                                            </TableCell>

                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </div>
                    ) :
                        <div className="text-muted-foreground mt-2 text-xs">
                            {pollingEnabled ? (
                                <span>处理完成后会显示检测结果。</span>
                            ) : (
                                <span>暂无检测结果，可以在下方运行 AI 检测。</span>
                            )}
                        </div>
                    }


                    {/* Bottom section */}
                    <div className="w-full mt-2">
                        {pollingEnabled ? (
                            <div className="w-full">

                                {detectionStatus.data !== undefined && (
                                    <>
                                        <Progress value={detectionStatus.data.progress} />
                                        <p className="text-sm text-muted-foreground mt-1">
                                            {detectionStatus.data.message ? <>{detectionStatus.data.message}</> : <>{detectionStatus.data.status}</>} — {Math.round(detectionStatus.data.progress)}%
                                        </p>
                                    </>
                                )}
                            </div>
                        ) : (
                            <div className="w-full flex flex-col">

                                <div className="w-full flex flex-row justify-between items-center">

                                    <Select
                                        value={detectionMode}
                                        onValueChange={(value) => setDetectionMode(value as "fast" | "medium" | "detailed" | "experimental" | undefined)}
                                    >
                                        <SelectTrigger className="w-[150px]"
                                            value={detectionMode}
                                        >
                                            <SelectValue placeholder="分析模式" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="fast">快速</SelectItem>
                                            <SelectItem value="medium">标准</SelectItem>
                                            <SelectItem value="detailed">精细</SelectItem>
                                            <SelectItem value="experimental">高精度 YOLO</SelectItem>
                                        </SelectContent>
                                    </Select>

                                    <div className="flex items-center gap-2">
                                        {hasDetections && hasVoronoi && (
                                            <Tooltip>
                                                <TooltipTrigger asChild>
                                                    <Button
                                                        variant={clipDetections ? "outline": "default"}
                                                        size="sm"
                                                        onClick={() => setClipDetections(!clipDetections)}
                                                    >
                                                        <Scissors className="w-4 h-4 mr-1" />
                                                        {clipDetections ? "显示全部" : "裁剪重复"}
                                                    </Button>
                                                </TooltipTrigger>
                                                <TooltipContent>
                                                    {clipDetections
                                                        ? "当前按每张图中心区域裁剪检测结果"
                                                        : "当前显示全部检测结果，重叠区域可能有重复目标"}
                                                </TooltipContent>
                                            </Tooltip>
                                        )}
                                        <Tooltip>
                                            <TooltipTrigger>
                                                <Button variant={`${detectionMode === undefined ? "outline" : "default"}`} size="sm" onClick={() => { handleStart() }} disabled={!pollingEnabled && (!detectionMode)}>
                                                    运行检测
                                                </Button>
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                {detectionMode === undefined ? "请先选择分析模式" : "开始 AI 目标检测"}
                                            </TooltipContent>
                                        </Tooltip>
                                    </div>
                                </div>
                                {detectionMode && (
                                    <div className="rounded-md border p-2 mt-2 text-sm border-gray-400 bg-gray-200 text-muted-foreground dark:bg-gray-800 dark:border-gray-700">
                                        <p className="m-0">
                                            <Info className="inline-block w-3 h-3 align-middle mr-1" />
                                            {infotextForMode(detectionMode)}
                                        </p>
                                    </div>

                                )}
                            </div>
                        )}

                    </div>
                </CardContent>
            </Card>
        </>
    );

}
